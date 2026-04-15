use std::{
    borrow::Cow,
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

use byteorder::{ReadBytesExt, WriteBytesExt};
use rkyv::{api::high::to_bytes_in, rancor, util::AlignedVec, with::AsOwned};

use crate::system::Coords;

#[derive(rkyv::Serialize, rkyv::Deserialize, rkyv::Archive)]
pub struct OutputSystem {
    pub name: String,
    pub coords: Coords,
    pub refuelable: bool,
    pub neutron_star: bool,
}

// Used only for serialization. `name` borrows directly from the line buffer
// via Cow::Borrowed, so no String allocation is needed. The #[rkyv(with =
// AsOwned)] attribute makes Cow<str> archive to ArchivedString — identical
// binary layout to OutputSystem — so GalaxyArchiveIter can read both.
#[derive(rkyv::Archive, rkyv::Serialize)]
struct OutputSystemRef<'a> {
    #[rkyv(with = AsOwned)]
    name: Cow<'a, str>,
    coords: Coords,
    refuelable: bool,
    neutron_star: bool,
}

/// Iterator over a `galaxy.rkyv` archive file produced by [`parse_galaxy_json`].
///
/// Each call to [`next`](GalaxyArchiveIter::next) returns a reference to the
/// zero-copy [`rkyv::Archived<OutputSystem>`] view of the next record,
/// borrowing from an internal buffer that is reused on every call.
/// Use a `while let` loop rather than `for`:
///
/// ```rust,ignore
/// let mut iter = GalaxyArchiveIter::new("galaxy.rkyv")?;
/// while let Some(sys) = iter.next() {
///     println!("{}", sys.name);
/// }
/// ```
pub struct GalaxyArchiveIter {
    pub reader: std::io::BufReader<File>,
    pub buf: Vec<u8>,
}

impl GalaxyArchiveIter {
    pub fn new(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: std::io::BufReader::new(file),
            buf: Vec::new(),
        })
    }

    pub fn next(&mut self) -> Option<&ArchivedOutputSystem> {
        let len = self.reader.read_u16::<byteorder::LE>().ok()? as usize;
        self.buf.resize(len, 0);
        self.reader.read_exact(&mut self.buf).ok()?;
        rkyv::access::<ArchivedOutputSystem, rancor::Error>(&self.buf).ok()
    }
}

// Parses the galaxy json file to find neutron stars that are close to the main star
// galaxy.json.gz is an absolutely huge file (100 GB) so pray that you don't need to run this more than once
pub fn parse_galaxy_json<R: Read, W: Write>(in_json: R, out_rkyv: W) -> io::Result<()> {
    let mut buf_reader = std::io::BufReader::with_capacity(8 * 1024 * 1024, in_json); // 8 MB buffer

    let mut buf_writer = std::io::BufWriter::with_capacity(1024 * 1024, out_rkyv); // 1 MB buffer

    #[derive(serde::Deserialize, serde::Serialize)]
    struct JsonCoords {
        x: f32,
        y: f32,
        z: f32,
    }

    impl From<JsonCoords> for Coords {
        fn from(c: JsonCoords) -> Self {
            Coords([c.x, c.y, c.z])
        }
    }

    // Derived booleans from the bodies array, computed inline without a Vec.
    #[derive(Default)]
    struct BodyDetails {
        main_is_neutron: bool,
        nearby_neutron_star: bool,
        main_star_refuelable: bool,
        has_population: bool,
    }

    // Deserialized one at a time on the stack inside the visitor; never collected.
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Body<'a> {
        sub_type: Option<&'a str>,
        distance_to_arrival: Option<f32>,
        #[serde(default)]
        main_star: bool,
        population: Option<u64>,
    }

    fn deserialize_bodies<'de, D: serde::Deserializer<'de>>(
        de: D,
    ) -> Result<BodyDetails, D::Error> {
        use serde::de::{SeqAccess, Visitor};

        struct BodiesVisitor;

        impl<'de> Visitor<'de> for BodiesVisitor {
            type Value = BodyDetails;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "an array of bodies")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<BodyDetails, A::Error> {
                let mut summary = BodyDetails::default();

                while let Some(body) = seq.next_element::<Body<'de>>()? {
                    if body.main_star {
                        summary.main_is_neutron = body.sub_type == Some("Neutron Star");
                        summary.main_star_refuelable = body
                            .sub_type
                            .is_some_and(|t| t.starts_with(['K', 'G', 'B', 'F', 'O', 'A', 'M']));
                    }
                    if body.sub_type == Some("Neutron Star")
                        && body.distance_to_arrival.unwrap_or(0.0) <= 3000.0
                    {
                        summary.nearby_neutron_star = true;
                    }
                    if body.population.unwrap_or(0) > 0 {
                        summary.has_population = true;
                    }
                }
                Ok(summary)
            }
        }

        de.deserialize_seq(BodiesVisitor)
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct System<'a> {
        name: &'a str,
        coords: JsonCoords,
        #[serde(deserialize_with = "deserialize_bodies")]
        bodies: BodyDetails,
    }

    let mut buf: Vec<u8> = Vec::new();
    buf_reader.skip_until(b'\n')?; // skip first line
    let mut i = 0;
    let mut rkyv_buf: AlignedVec<16> = AlignedVec::new();

    let mut deadzone_n = 0;

    while let Ok(bytes_read) = buf_reader.read_until(b'\n', &mut buf) {
        if bytes_read == 0 {
            break; // EOF
        }
        i += 1;
        buf.pop(); // remove '\n'
        if buf.last() == Some(&b']') {
            break;
        }
        if buf.last() == Some(&b',') {
            buf.pop();
        }
        let system = match simd_json::from_slice::<System>(&mut buf) {
            Ok(sys) => sys,
            Err(e) => panic!("Failed to parse JSON on line {}: {}", i, e),
        };

        let bodies = &system.bodies;
        let specially_named = !system.name.contains('-'); // rough heuristic, custom named systems often have no dashes
        let inside_deadzone = system.coords.x.abs() <= 800.0
            || system.coords.y.abs() <= 800.0
            || system.coords.z.abs() <= 800.0;

        // sample systems in the deadzone, where there are many stars but few neutrons
        let chosen_in_deadzone = inside_deadzone && deadzone_n % 256 == 0;

        if inside_deadzone {
            deadzone_n += 1;
        }

        if bodies.main_is_neutron
            || (bodies.nearby_neutron_star && bodies.main_star_refuelable)
            || bodies.has_population
            || specially_named
            || chosen_in_deadzone
        {
            rkyv_buf.clear();
            rkyv_buf = to_bytes_in::<_, rancor::Error>(
                &OutputSystemRef {
                    name: Cow::Borrowed(system.name),
                    coords: system.coords.into(),
                    refuelable: bodies.main_star_refuelable,
                    neutron_star: bodies.nearby_neutron_star,
                },
                rkyv_buf,
            )
            .unwrap();
            buf_writer
                .write_u16::<byteorder::LE>(rkyv_buf.len() as u16)
                .unwrap();
            buf_writer.write_all(&rkyv_buf).unwrap();
        }

        buf.clear();
    }

    Ok(())
}
