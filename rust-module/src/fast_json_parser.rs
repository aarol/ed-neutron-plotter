use memchr::memchr;

use crate::system::Coords;

pub fn parse_line(line: &str) -> Option<(String, Coords)> {
    // Skip empty lines and closing/opening bracket
    if line.is_empty() || line == "]" || line == "[" {
        return None;
    }

    // Remove trailing comma if present
    let line = line.trim_end_matches(',');

    let bytes = line.as_bytes();

    // Find "name":" and extract the name value
    let name = parse_str(b"\"name\":\"", bytes).expect("Name found").to_string();
    // Find coords
    let coords = {
        let coords_key = b"\"coords\":{";
        let coords_start = memchr::memmem::find(bytes, coords_key)? + coords_key.len();

        let x = parse_f32(b"\"x\":", &bytes[coords_start..], b',').expect("X found");
        let y = parse_f32(b"\"y\":", &bytes[coords_start..], b',').expect("Y found");
        let z = parse_f32(b"\"z\":", &bytes[coords_start..], b'}').expect("Z found");
        Coords::new(x, y, z)
    };

    Some((name, coords))
}

fn parse_f32(needle: &[u8], haystack: &[u8], end_char: u8) -> Option<f32> {
    let x_start = memchr::memmem::find(haystack, needle)? + needle.len();
    let x_end = memchr(end_char, &haystack[x_start..])?;
    let x_str = unsafe { std::str::from_utf8_unchecked(&haystack[x_start..x_start + x_end]) };
    Some(x_str.parse::<f32>().unwrap_or_else(|_| panic!("{} Valid", x_str)))
}

fn parse_str<'a>(needle: &[u8], haystack: &'a [u8]) -> Option<&'a str> {
    let start = memchr::memmem::find(&haystack, needle)? + needle.len();
    let end = memchr(b'"', &haystack[start..])?;
    let str_slice = unsafe { std::str::from_utf8_unchecked(&haystack[start..start + end]) };
    Some(str_slice)
}
