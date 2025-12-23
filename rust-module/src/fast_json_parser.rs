use memchr::memchr;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};

use crate::system::Coords;

pub struct SystemParser<R: Read> {
    reader: BufReader<R>,
    buffer: String,
}

impl<R: Read + Seek> SystemParser<R> {
    pub fn new(mut reader: R) -> std::io::Result<Self> {
        // Skip the opening bracket
        reader.seek(SeekFrom::Start(1))?;

        Ok(Self {
            reader: BufReader::new(reader),
            buffer: String::with_capacity(512),
        })
    }

    pub fn for_each<F>(mut self, mut f: F) -> std::io::Result<()>
    where
        F: FnMut(&str, Coords),
    {
        let mut first_line = true;

        loop {
            self.buffer.clear();

            match self.reader.read_line(&mut self.buffer) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    // Skip the first line (we already seeked past the '[')
                    if first_line {
                        first_line = false;
                        continue;
                    }

                    if let Some((name, coords)) = Self::parse_line(&self.buffer) {
                        f(name, coords);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    fn parse_line(buffer: &str) -> Option<(&str, Coords)> {
        let line = buffer.trim();

        // Skip empty lines and closing bracket
        if line.is_empty() || line == "]" {
            return None;
        }

        // Remove trailing comma if present
        let line = line.trim_end_matches(',');

        let bytes = line.as_bytes();

        // Find "name":" and extract the name value
        let name = {
            let name_key = b"\"name\":\"";
            let name_start = memchr::memmem::find(bytes, name_key)? + name_key.len();
            let name_end = memchr(b'"', &bytes[name_start..])?;

            // SAFETY: We're taking a slice of the original line string at valid UTF-8 boundaries
            // since we found the quote characters which are ASCII
            unsafe { std::str::from_utf8_unchecked(&bytes[name_start..name_start + name_end]) }
        };

        // Find coords
        let coords = {
            let coords_key = b"\"coords\":{";
            let coords_start = memchr::memmem::find(bytes, coords_key)? + coords_key.len();

            // Find x
            let x_key = b"\"x\":";
            let x_start =
                memchr::memmem::find(&bytes[coords_start..], x_key)? + coords_start + x_key.len();
            let x_end = memchr(b',', &bytes[x_start..])?;
            let x_str = unsafe { std::str::from_utf8_unchecked(&bytes[x_start..x_start + x_end]) };
            let x = x_str.parse::<f32>().ok()?;

            // Find y
            let y_key = b"\"y\":";
            let y_start =
                memchr::memmem::find(&bytes[coords_start..], y_key)? + coords_start + y_key.len();
            let y_end = memchr(b',', &bytes[y_start..])?;
            let y_str = unsafe { std::str::from_utf8_unchecked(&bytes[y_start..y_start + y_end]) };
            let y = y_str.parse::<f32>().ok()?;

            // Find z
            let z_key = b"\"z\":";
            let z_start =
                memchr::memmem::find(&bytes[coords_start..], z_key)? + coords_start + z_key.len();
            let z_end = memchr(b'}', &bytes[z_start..])?;
            let z_str = unsafe { std::str::from_utf8_unchecked(&bytes[z_start..z_start + z_end]) };
            let z = z_str.parse::<f32>().ok()?;

            Coords::new(x, y, z)
        };

        Some((name, coords))
    }
}
