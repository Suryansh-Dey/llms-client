use crate::utils::*;

#[test]
fn image_detection_regex() {
    let image_regex = get_image_regex();
    let images =
        image_regex.captures_iter(" water is good ![but fire](https://looks-good). thanks thanks");
    for image in images {
        assert_eq!(&image[1], "but fire");
        assert_eq!(&image[2], "https://looks-good");
    }
}
