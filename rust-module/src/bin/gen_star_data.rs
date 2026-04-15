use mimalloc::MiMalloc;

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::{
    collections::BTreeSet,
    fs::File,
    io::{self, BufWriter, Read, Write},
};

use rust_module::{
    galaxy_parser::{self},
    kdtree,
    system::{Coords, System},
    trie::LoudsTrie,
};

#[derive(rkyv::Archive, rkyv::Deserialize, Debug, rkyv::Serialize, PartialEq)]
#[rkyv(compare(PartialEq))]
struct SystemData {
    systems: Vec<System>,
}

fn main() -> io::Result<()> {
    if std::env::args().any(|arg| arg.contains("analyze")) {
        return analyze_trie();
    }

    // Reads the galaxy.json file from stdin, parses it, and writes the rkyv archive to galaxy.rkyv.
    if std::env::args().any(|arg| arg.contains("parse_galaxy")) {
        let stdin = io::stdin();
        let stdin_handle = stdin.lock();
        let output = File::create("galaxy.rkyv")?;
        return galaxy_parser::parse_galaxy_json(stdin_handle, output);
    }

    process_galaxy_rkyv()?; // This is the main code path for generating the binary data files from the rkyv archive
    Ok(())
}

fn process_galaxy_rkyv() -> io::Result<()> {
    let out_dir = std::path::Path::new("../public/data");
    std::fs::create_dir_all(out_dir)?;

    let galaxy_file = File::open("galaxy.rkyv").inspect_err(|_| {
        println!("Error: galaxy.rkyv not found. You need to generate it first. See the README.md for instructions.")
    })?;

    let mut system_set = BTreeSet::new();

    let mut iter = galaxy_parser::GalaxyArchiveIter {
        reader: std::io::BufReader::new(galaxy_file),
        buf: Vec::new(),
    };

    let mut i = 0;
    while let Some(system) = iter.next() {
        if i % 100000 == 0 {
            println!("Processing line {}", i);
        }
        let coords = &system.coords;
        // The coordinates are in light years (+-30k), three.js doesn't like such huge distances
        // This will reduce the scale to max [-100, 100] in each axis
        // the EDSM api also uses this scale
        let coords = Coords::new(
            -(coords.0[0] / 1000.0),
            coords.0[1] / 1000.0,
            coords.0[2] / 1000.0,
        );

        system_set.insert(System {
            name: system.name.to_string(),
            coords,
            is_neutron_star: system.neutron_star,
            searchable: true,
        });
        i += 1;
    }
    println!("Parsing complete. Contructing the trie..");
    // Create (name, original_index) pairs and sort by name
    // so that we can remap star coordinates from original order to trie order
    // this way, the trie does not need to store the coordinate indices explicitly
    let str_keys: Vec<&str> = system_set
        .iter()
        .filter_map(|system| system.searchable.then(|| system.name.as_str()))
        .collect();

    assert!(str_keys.is_sorted());

    let mut trie_file = File::create(out_dir.join("search_trie.bin"))?;
    let mut trie_file_buf = BufWriter::new(&mut trie_file);
    let coords_indices = LoudsTrie::build(&str_keys, &mut trie_file_buf)?;
    trie_file_buf.flush()?;
    drop(trie_file_buf);

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
    let mut trie_buf = vec![];
    File::open(out_dir.join("search_trie.bin"))?.read_to_end(&mut trie_buf)?;
    let trie = LoudsTrie::from_bytes(&trie_buf);

    for star in trie.suggest("Speam", 10) {
        println!("  - {}: {:?}", star, trie.find(&star));
    }

    println!("\nTrie has {} nodes", trie.node_count());

    println!(
        "Uses {:.1} MB of space",
        trie.size_on_disk() as f64 / 1024.0 / 1024.0
    );

    Ok(())
}

#[allow(dead_code)]
fn analyze_trie() -> io::Result<()> {
    let mut file = std::io::BufReader::new(File::open("../public/data/search_trie.bin")?);

    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    let trie = LoudsTrie::from_bytes(&buf);
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
