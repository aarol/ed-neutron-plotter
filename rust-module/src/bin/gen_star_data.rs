use std::{
    collections::BTreeSet,
    io::{self, Read, Write},
};

use rust_module::{fast_json_parser::SystemParser, kdtree, system::{Coords, System}, trie::LoudsTrie};

#[allow(dead_code)]
fn analyze() -> io::Result<()> {
    let mut file = std::io::BufReader::new(std::fs::File::open("../public/data/search_trie.bin")?);

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

fn main() -> io::Result<()> {
    if std::env::args().any(|arg| arg.contains("analyze")) {
        return analyze();
    }

    let out_dir = std::path::Path::new("../public/data");
    std::fs::create_dir_all(out_dir)?;

    let neutrons_file = std::fs::File::open("systems_neutron.json")?;
    let regular_systems_file = std::fs::File::open("systems_1day.json")?;

    let mut system_set = BTreeSet::new();

    let parser = SystemParser::new(neutrons_file)?;
    let parser2 = SystemParser::new(regular_systems_file)?;

    let mut count = 0;

    parser
    .chain(parser2)
    .for_each(|(name, coord)| {
        if count % 100000 == 0 {
            println!("Processing line {}", count);
        }
        count += 1;

        // The coordinates are in light years (+-30k), three.js doesn't like such huge distances
        // This will reduce the scale to max [-100, 100] in each axis
        // the EDSM api also uses this scale
        let coords = Coords::new(
            -(coord.x() / 1000.0),
            coord.y() / 1000.0,
            coord.z() / 1000.0,
        );

        system_set.insert(System {name: name.to_owned(), coords});
    });

    // Create (name, original_index) pairs and sort by name
    // so that we can remap star coordinates from original order to trie order
    // this way, the trie does not need to store the coordinate indices explicitly
    let str_keys: Vec<&str> = system_set
        .iter()
        .map(|system| system.name.as_str())
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
    let kdtree = kdtree::CompactKdTree::new(&kdtree_indices);

    let mut kdtree_file = std::fs::File::create(out_dir.join("star_kdtree.bin"))?;
    kdtree_file.write_all(&kdtree.to_bytes())?;

    let mut stars_file = std::fs::File::create(out_dir.join("neutron_stars0.bin"))?;

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
    let mut trie_file = std::fs::File::create(out_dir.join("search_trie.bin"))?;
    let trie_bytes: Vec<u8> = trie.into();
    trie_file.write_all(&trie_bytes)?;

    Ok(())
}
