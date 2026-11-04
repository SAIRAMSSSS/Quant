pub const PAGE_SIZE: usize = 4096;

#[inline]
pub fn align_to_page(offset: usize, len: usize) -> (usize, usize) {
    if len == 0 {
        return (offset & !(PAGE_SIZE - 1), 0);
    }
    let start = offset & !(PAGE_SIZE - 1);
    let end = (offset.saturating_add(len).saturating_add(PAGE_SIZE - 1)) & !(PAGE_SIZE - 1);
    (start, end.saturating_sub(start))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_offset_exact_page() {
        let (start, len) = align_to_page(0, 4096);
        assert_eq!(start, 0);
        assert_eq!(len, 4096);
    }

    #[test]
    fn test_mid_page_single() {
        let (start, len) = align_to_page(100, 500);
        assert_eq!(start, 0);
        assert_eq!(len, 4096);
    }

    #[test]
    fn test_mid_page_cross_boundary() {
        let (start, len) = align_to_page(4000, 200);
        assert_eq!(start, 0);
        assert_eq!(len, 8192);
    }

    #[test]
    fn test_aligned_offset_multi_page() {
        let (start, len) = align_to_page(8192, 12288);
        assert_eq!(start, 8192);
        assert_eq!(len, 12288);
    }

    #[test]
    fn test_zero_length() {
        let (start, len) = align_to_page(4500, 0);
        assert_eq!(start, 4096);
        assert_eq!(len, 0);
    }
}
