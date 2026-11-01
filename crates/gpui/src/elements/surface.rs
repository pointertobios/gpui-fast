use std::sync::Arc;

use crate::{
    App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Length, ObjectFit, Pixels, Size, Style, StyleRefinement, Styled, Window,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc2_core_foundation::CFRetained;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc2_core_video::{CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetWidth};
use refineable::Refineable;

/// The colour a luma and chroma plane are in.
///
/// The renderer turns the samples into RGB on the GPU, so how to do that has to
/// travel with them: a video in the wrong matrix is a picture that looks a
/// little too green rather than one that does not draw at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YuvColor {
    /// The samples use the whole byte range rather than the video range.
    pub full_range: bool,
    /// The matrix coefficients, by their `nclx` number: 1 is BT.709, 5 and 6
    /// are BT.601, 9 and 10 are BT.2020.
    pub matrix: u16,
}

impl Default for YuvColor {
    fn default() -> Self {
        YuvColor {
            full_range: false,
            matrix: 1,
        }
    }
}

/// A picture's planes in system memory: luma, and the two chroma planes, at
/// their own size.
///
/// This is what a decoder hands back and what a renderer samples, and it is the
/// *only* representation a film passes through: converting it to RGB is the
/// renderer's to do on the GPU, which is what makes a picture of one byte and a
/// half per pixel the cost of drawing it — a third of the four a converted
/// picture is, and no conversion pass on the way there.
///
/// `id` names the surface rather than the picture, so a film that hands a new
/// set of planes thirty times a second keeps one pair of textures in the
/// renderer instead of making a pair a frame.
#[derive(Debug, PartialEq, Eq)]
pub struct Planes {
    /// Which surface this is a picture of: stable for a film, so its textures
    /// are reused.
    pub id: usize,
    /// The picture's width in pixels.
    pub width: u32,
    /// The picture's height in pixels.
    pub height: u32,
    /// Luma, `width * height` bytes, tightly packed.
    ///
    /// Shared rather than owned: a film hands over a new set of planes thirty
    /// times a second, and the window is one reader of them. A decoder that
    /// keeps its own buffers can hand out the same bytes, and a repaint that
    /// brings no new picture compares as the same allocation.
    pub y: Arc<[u8]>,
    /// Cb, `width / 2 * height / 2` bytes for 4:2:0, tightly packed.
    pub cb: Arc<[u8]>,
    /// Cr, the same size as [`Self::cb`].
    pub cr: Arc<[u8]>,
    /// The matrix and range the samples were encoded with.
    pub color: YuvColor,
    /// Which picture of the surface this is, counted by whoever decodes them.
    ///
    /// It is what tells a repaint that brings a new picture — one to upload —
    /// from one that brings the same picture again, which a hover or a drag of
    /// the bar does. A window that only compared the bytes would have to copy
    /// them to compare them.
    pub frame: u64,
}

impl Planes {
    /// The bytes of one picture, in the layout a renderer samples.
    pub fn new(
        id: usize,
        width: u32,
        height: u32,
        y: Arc<[u8]>,
        cb: Arc<[u8]>,
        cr: Arc<[u8]>,
    ) -> Planes {
        Planes {
            id,
            width,
            height,
            y,
            cb,
            cr,
            color: YuvColor::default(),
            frame: 0,
        }
    }

    /// This picture is the `frame`th of its surface.
    pub fn frame(mut self, frame: u64) -> Self {
        self.frame = frame;
        self
    }

    /// These planes in `color`'s matrix and range.
    pub fn color(mut self, color: YuvColor) -> Self {
        self.color = color;
        self
    }
}

/// A source of a surface's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceSource {
    /// A CoreVideo image buffer
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Surface(CFRetained<CVPixelBuffer>),
    /// Planes in system memory, for a renderer that samples them itself.
    Planes(Arc<Planes>),
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl From<CFRetained<CVPixelBuffer>> for SurfaceSource {
    fn from(value: CFRetained<CVPixelBuffer>) -> Self {
        SurfaceSource::Surface(value)
    }
}

impl From<Arc<Planes>> for SurfaceSource {
    fn from(value: Arc<Planes>) -> Self {
        SurfaceSource::Planes(value)
    }
}

/// A surface element.
pub struct Surface {
    source: SurfaceSource,
    object_fit: ObjectFit,
    style: StyleRefinement,
    /// A key the tests look this element's room up by. Without one, a caller has
    /// to wrap the surface to measure it — and then a surface laid out at
    /// nothing paints nothing while the box around it still measures as a
    /// picture, which is how a film comes to play with no picture on it.
    #[cfg(any(test, all(debug_assertions, feature = "test-support")))]
    debug_selector: Option<String>,
}

/// Create a new surface element.
pub fn surface(source: impl Into<SurfaceSource>) -> Surface {
    Surface {
        source: source.into(),
        object_fit: ObjectFit::Contain,
        style: Default::default(),
        #[cfg(any(test, all(debug_assertions, feature = "test-support")))]
        debug_selector: None,
    }
}

impl Surface {
    /// Set the object fit for the image.
    pub fn object_fit(mut self, object_fit: ObjectFit) -> Self {
        self.object_fit = object_fit;
        self
    }

    /// Set a key this surface's room can be looked up by in a test, the way a
    /// `div`'s can.
    ///
    /// Selectors are only recorded in debug builds with the `test-support`
    /// feature. Elsewhere this is a no-op.
    #[cfg(any(test, all(debug_assertions, feature = "test-support")))]
    pub fn debug_selector(mut self, f: impl FnOnce() -> String) -> Self {
        self.debug_selector = Some(f());
        self
    }

    /// Set a key this surface's room can be looked up by in a test.
    #[cfg(not(any(test, all(debug_assertions, feature = "test-support"))))]
    #[inline]
    pub fn debug_selector(self, _: impl FnOnce() -> String) -> Self {
        self
    }

    /// The size the source is drawn at when nothing else says: the picture's own
    /// pixels, and none for a source that cannot be measured.
    fn source_size(&self) -> Size<Pixels> {
        match &self.source {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            SurfaceSource::Surface(surface) => crate::size(
                CVPixelBufferGetWidth(surface).into(),
                CVPixelBufferGetHeight(surface).into(),
            ),
            SurfaceSource::Planes(planes) => crate::size(planes.width.into(), planes.height.into()),
            #[allow(unreachable_patterns)]
            _ => crate::size(Pixels::ZERO, Pixels::ZERO),
        }
    }
}

impl Element for Surface {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let mut style = Style::default();
        style.refine(&self.style);
        // A surface is a leaf with no children to be sized by, so a caller that
        // does not name a size would lay it out as nothing at all — an empty box
        // where the picture should be. The picture's own size is what it is
        // drawn at when nothing says otherwise, the same way an image element
        // is sized by the image.
        let size = self.source_size();
        if matches!(style.size.width, Length::Auto) {
            style.size.width = Length::Definite(size.width.into());
        }
        if matches!(style.size.height, Length::Auto) {
            style.size.height = Length::Definite(size.height.into());
        }
        let layout_id = window.request_layout(style, [], cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
    }

    fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        _: &mut App,
    ) {
        match &self.source {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            SurfaceSource::Surface(surface) => {
                let size = crate::size(
                    CVPixelBufferGetWidth(surface).into(),
                    CVPixelBufferGetHeight(surface).into(),
                );
                let new_bounds = self.object_fit.get_bounds(bounds, size);
                // TODO: Add support for corner_radii
                window.paint_surface(new_bounds, surface.clone());
            }
            // A renderer with an image buffer of its own draws that; the
            // planes are for one that samples them itself, which is every
            // platform whose window is not a Metal one.
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            SurfaceSource::Planes(planes) => {
                let size = crate::size(planes.width.into(), planes.height.into());
                let new_bounds = self.object_fit.get_bounds(bounds, size);
                #[cfg(any(test, all(debug_assertions, feature = "test-support")))]
                if let Some(selector) = &self.debug_selector {
                    window
                        .next_frame
                        .record_debug_bounds(selector.clone(), new_bounds);
                }
                window.paint_planes(new_bounds, planes.clone());
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

impl IntoElement for Surface {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Surface {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}
