use std::{
    collections::BTreeSet,
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

use memmap::MmapOptions;
use rkyv::rancor;
use rust_module::{
    fast_json_parser::parse_line,
    kdtree,
    system::{Coords, System},
    trie::LoudsTrie,
};

#[allow(dead_code)]
fn analyze_trie() -> io::Result<()> {
    let mut file = std::io::BufReader::new(File::open("../public/data/search_trie.bin")?);

    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    let trie = LoudsTrie::from(buf.as_ref());
    // println!("Trie has {} nodes", trie());

    // Test contains
    println!("\nContains 'Colonia': {}", trie.find("Colonia").is_some());

    // Test suggest with different prefixes
    println!("\nSuggestions for 'Col':");
    for suggestion in trie.suggest("Col", 10) {
        println!("  - {}", suggestion);
    }

    println!("\nSuggestions for 'Colonia':");
    for suggestion in trie.suggest("Colonia", 10) {
        println!("  - {}", suggestion);
    }

    println!("\nSuggestions for 'Speam':");
    for suggestion in trie.suggest("Speam", 10) {
        println!("  - {}", suggestion);
    }

    println!("\nSuggestions for 'Jackson:");
    for suggestion in trie.suggest("Jackson", 10) {
        println!("  - {}", suggestion);
    }

    trie.analyze_structure();

    Ok(())
}

#[derive(rkyv::Archive, rkyv::Deserialize, Debug, rkyv::Serialize, PartialEq)]
#[rkyv(compare(PartialEq))]
struct SystemData {
    systems: Vec<System>,
}

fn parse_systems_rkyv() -> io::Result<()> {
    let mut neutrons_file = File::open("systems_neutron.json").unwrap();
    let buf_reader = std::io::BufReader::new(&mut neutrons_file);
    println!("Processing systems_neutron.json..");

    let mut systems: Vec<System> = buf_reader
        .lines()
        .skip(1)
        .flat_map(|line| parse_line(&line.unwrap()))
        .enumerate()
        .map(|(i, (name, coords))| {
            if i % 100000 == 0 {
                println!("Processing line {}", i);
            }
            System {
                name: name.to_string(),
                coords,
                is_neutron_star: true,
                searchable: true,
            }
        })
        .collect();
    let mut systems_file = File::open("systems_1day.json").unwrap();
    // let mut gz_decoder = flate2::read::GzDecoder::new(&mut systems_file);
    let buf_reader = std::io::BufReader::new(&mut systems_file);

    // Include systems_1day.json 
    // Some of these will be neutron stars already included above, but that's fine
    // because duplicates will be removed later when constructing the btree set
    println!("Processing systems_1day.json..");
    let mut res: Vec<System> = buf_reader
        .lines()
        .skip(1)
        // .step_by(20)
        .flat_map(|line| parse_line(&line.unwrap()))
        .enumerate()
        .filter_map(|(i, (name, coords))| {
            if i % 100000 == 0 {
                println!("Processing line {}", i);
            }
            // if coords.x().abs() <= 800.0 || coords.z().abs() <= 800.0 {
                Some(System {
                    name,
                    coords,
                    is_neutron_star: false,
                    searchable: true,
                })
            // }
            // return None;
        })
        .collect();

    systems.append(&mut res);

    let system_data = SystemData { systems };
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&system_data).unwrap();

    let mut out_file = File::create("systems.rkyv")?;
    out_file.write_all(&bytes)?;

    Ok(())
}

fn main() -> io::Result<()> {
    if std::env::args().any(|arg| arg.contains("analyze")) {
        return analyze_trie();
    }

    if std::env::args().any(|arg| arg.contains("parse_systems")) {
        return parse_systems_rkyv();
    }

    if !Path::new("systems.rkyv").exists() {
        println!("systems.rkyv not found, generating it first..");
        parse_systems_rkyv()?;
    }

    let out_dir = std::path::Path::new("../public/data");
    std::fs::create_dir_all(out_dir)?;

    let neutrons_archive = File::open("systems.rkyv")?;
    let mmap = unsafe { MmapOptions::new().map(&neutrons_archive)? };

    let mut system_set = BTreeSet::new();
    let archived_system_data = rkyv::access::<ArchivedSystemData, rancor::Error>(&mmap).unwrap();

    archived_system_data
        .systems
        .iter()
        .enumerate()
        .for_each(|(i, sys)| {
            if i % 100000 == 0 {
                println!("Processing line {}", i);
            }
            let coords = &sys.coords;
            // The coordinates are in light years (+-30k), three.js doesn't like such huge distances
            // This will reduce the scale to max [-100, 100] in each axis
            // the EDSM api also uses this scale
            let coords = Coords::new(
                -(coords.0[0] / 1000.0),
                coords.0[1] / 1000.0,
                coords.0[2] / 1000.0,
            );

            system_set.insert(System {
                name: sys.name.to_string(),
                coords,
                is_neutron_star: sys.is_neutron_star,
                searchable: sys.searchable,
            });
        });
    println!("Parsing complete. Contructing the trie..");
    // Create (name, original_index) pairs and sort by name
    // so that we can remap star coordinates from original order to trie order
    // this way, the trie does not need to store the coordinate indices explicitly
    let str_keys: Vec<&str> = system_set
        .iter()
        .filter_map(|system| system.searchable.then(|| system.name.as_str()))
        .collect();
    let (trie, coords_indices) = LoudsTrie::new(&str_keys);

    let star_coords: Vec<[f32; 3]> = system_set.iter().map(|s| s.coords.to_slice()).collect();

    // Map from trie order to original coord order
    println!("Reassigning star coords according to trie..");
    let mut sorted_coords = vec![[0.0; 3]; star_coords.len()];
    for (sorted_idx, &trie_idx) in coords_indices.iter().enumerate() {
        sorted_coords[trie_idx] = star_coords[sorted_idx];
    }

    let kdtree_indices = kdtree::KdTreeBuilder::from_points(&sorted_coords).build();
    let kdtree = kdtree::CompactKdTree::new(kdtree_indices.into_boxed_slice());

    let mut kdtree_file = File::create(out_dir.join("star_kdtree.bin"))?;
    kdtree_file.write_all(&kdtree.to_bytes())?;

    let mut stars_file = File::create(out_dir.join("neutron_stars0.bin"))?;

    let bytes = unsafe {
        std::slice::from_raw_parts(
            sorted_coords.as_ptr() as *const u8,
            sorted_coords.len() * std::mem::size_of::<Coords>(),
        )
    };
    stars_file.write_all(bytes)?;
    // partitions.par_iter().enumerate().for_each(|(i, p)| {
    //     let mut file =
    //         std::fs::File::create(out_dir.join(format!("neutron_stars{}.bin", i))).unwrap();

    //     p.write_to_file(&mut file).unwrap();
    // });

    for star in trie.suggest("Speam", 10) {
        println!("  - {}: {:?}", star, trie.find(&star));
    }

    println!("\nTrie has {} nodes", trie.node_count());

    println!(
        "Uses {:.1} MB of space",
        trie.size_on_disk() as f64 / 1024.0 / 1024.0
    );

    // Write trie to file
    let mut trie_file = File::create(out_dir.join("search_trie.bin"))?;
    let trie_bytes: Vec<u8> = trie.into();
    trie_file.write_all(&trie_bytes)?;

    Ok(())
}
