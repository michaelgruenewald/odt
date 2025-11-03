use crate::path::NodePath;
use core::fmt::{Display, Formatter, Write};
use hashlink::linked_hash_map::Entry;
use hashlink::{LinkedHashMap, LinkedHashSet};

/// An intermediate representation of a node tree, used to gather deletes and overrides.
/// Preserves the input ordering while allowing random access.
///
/// Behavior does not match `dtc` in all cases.  Deleting a node and redefining it with
/// the same name will move it to the end, but `dtc` remembers the original ordering.
#[derive(Clone)]
pub struct Node<P> {
    labels: LinkedHashSet<String>,
    properties: LinkedHashMap<String, P>,
    children: LinkedHashMap<String, Node<P>>,
}

impl<P> Node<P> {
    pub fn walk<'a, 'b>(&'a self, path: impl IntoIterator<Item = &'b str>) -> Option<&'a Node<P>> {
        let mut path = path.into_iter();
        match path.next() {
            None | Some("") => Some(self),
            Some(segment) => self.children.get(segment)?.walk(path),
        }
    }

    pub fn walk_mut<'a, 'b>(
        &'a mut self,
        path: impl IntoIterator<Item = &'b str>,
    ) -> Option<&'a mut Node<P>> {
        let mut path = path.into_iter();
        match path.next() {
            None | Some("") => Some(self),
            Some(segment) => self.children.get_mut(segment)?.walk_mut(path),
        }
    }

    pub fn add_child(&mut self, name: &str) -> &mut Node<P> {
        // Avoid `Entry::or_insert_with()` because on LinkedHashMap that reorders existing entries.
        match self.children.entry(name.into()) {
            Entry::Vacant(entry) => entry.insert(Default::default()),
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }

    pub fn get_child_mut(&mut self, name: &str) -> Option<&mut Node<P>> {
        self.children.get_mut(name)
    }

    pub fn get_child(&self, name: &str) -> Option<&Node<P>> {
        self.children.get(name)
    }

    pub fn remove_child(&mut self, name: &str) {
        self.children.remove(name);
    }

    pub fn get_property(&self, name: &str) -> Option<&P> {
        self.properties.get(name)
    }

    pub fn set_property(&mut self, name: &str, value: P) -> Option<P> {
        self.properties.replace(name.into(), value)
    }

    pub fn remove_property(&mut self, name: &str) -> Option<P> {
        self.properties.remove(name)
    }

    pub fn children(&self) -> impl Iterator<Item = (&String, &Self)> {
        self.children.iter()
    }

    pub fn children_mut(&mut self) -> impl Iterator<Item = (&String, &mut Self)> {
        self.children.iter_mut()
    }

    pub fn properties(&self) -> impl Iterator<Item = (&String, &P)> {
        self.properties.iter()
    }

    pub fn properties_mut(&mut self) -> impl Iterator<Item = (&String, &mut P)> {
        self.properties.iter_mut()
    }

    pub fn add_label(&mut self, name: &str) {
        self.labels.replace(name.into());
    }

    pub fn labels(&mut self) -> impl Iterator<Item = &String> {
        self.labels.iter()
    }

    pub fn labels_as_display(&self) -> LabelsDisplay<'_> {
        LabelsDisplay(&self.labels)
    }

    pub fn map_values<T>(self, f: &mut impl FnMut(P) -> T) -> Node<T> {
        let Self {
            labels,
            properties,
            children,
        } = self;
        let properties = properties
            .into_iter()
            .map(|(k, v)| (k, f(v)))
            .collect::<LinkedHashMap<String, T>>();
        let children = children
            .into_iter()
            .map(|(k, v)| (k, v.map_values(f)))
            .collect::<LinkedHashMap<String, Node<T>>>();
        Node::<T> {
            labels,
            properties,
            children,
        }
    }

    /// Like `map_values()`, but provide the callback with the node path.
    pub fn map_located_values<T>(
        self,
        loc: &NodePath,
        f: &mut impl FnMut(&NodePath, P) -> T,
    ) -> Node<T> {
        let Self {
            labels,
            properties,
            children,
        } = self;
        let properties = properties
            .into_iter()
            .map(|(k, v)| (k, f(loc, v)))
            .collect::<LinkedHashMap<String, T>>();
        let children = children
            .into_iter()
            .map(|(k, v)| (k.clone(), v.map_located_values(&loc.join(&k), f)))
            .collect::<LinkedHashMap<String, Node<T>>>();
        Node::<T> {
            labels,
            properties,
            children,
        }
    }

    /// Sort child nodes and properties by name, recursively.
    pub fn sort(&mut self) {
        let mut keys: Vec<String> = self.properties.keys().cloned().collect();
        keys.sort_unstable();
        for k in keys {
            self.properties.to_back(&k);
        }
        let mut keys: Vec<String> = self.children.keys().cloned().collect();
        keys.sort_unstable();
        for k in keys {
            self.children.to_back(&k).map(Self::sort);
        }
    }
}

impl<P> Default for Node<P> {
    fn default() -> Self {
        Self {
            labels: Default::default(),
            properties: Default::default(),
            children: Default::default(),
        }
    }
}

pub trait OptionDisplay {
    fn fmt_opt(&self) -> Option<String>;
}

impl OptionDisplay for &crate::parse::rules::Prop<'_> {
    fn fmt_opt(&self) -> Option<String> {
        use crate::parse::TypedRuleExt;
        self.prop_value.map(|pv| pv.str().into())
    }
}

impl OptionDisplay for Vec<u8> {
    fn fmt_opt(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut f = String::new();
        // Formatting here is sloppy because we pass the output through the pretty-printer.
        if let Some(s) = guess_c_strings(self) {
            for (i, w) in s.split_terminator('\0').enumerate() {
                if i > 0 {
                    _ = write!(f, ", ");
                }
                _ = write!(f, "\"");
                for b in w.as_bytes() {
                    _ = write!(f, "{}", std::ascii::escape_default(*b));
                }
                _ = write!(f, "\"");
            }
        } else if self.len() % 4 == 0 {
            _ = write!(f, "<");
            for w in self.chunks_exact(4) {
                // stabilization of slice::array_chunks would simplify this
                let n = u32::from_be_bytes(w.try_into().unwrap());
                _ = write!(f, " {n:#04x}");
            }
            _ = write!(f, ">")
        } else {
            _ = write!(f, "[");
            for b in self {
                _ = write!(f, " {b:02x}");
            }
            _ = write!(f, "]");
        }
        Some(f)
    }
}

fn guess_c_strings(s: &[u8]) -> Option<&str> {
    if s.last() != Some(&0) {
        return None;
    }
    if s[0] == 0 && s.len() == 4 {
        // Don't interpret a cell value such as `0x00abcd00` as two strings.
        return None;
    }
    let mut nuls = 0;
    for c in s {
        match c {
            0 => nuls += 1,
            b' '..=b'~' => (),
            _ => return None,
        }
    }
    if nuls == 1 || nuls <= s.len() / 2 {
        Some(core::str::from_utf8(s).unwrap())
    } else {
        None
    }
}

impl<P: OptionDisplay> Display for Node<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "{{")?;
        for (name, value) in &self.properties {
            if let Some(value) = value.fmt_opt() {
                writeln!(f, "{name} = {value};")?;
            } else {
                writeln!(f, "{name};")?;
            }
        }
        if !self.properties.is_empty() && !self.children.is_empty() {
            writeln!(f)?;
        }
        for (name, node) in &self.children {
            writeln!(f, "{}{name} {node};", node.labels_as_display())?;
        }
        write!(f, "}}")
    }
}

pub struct LabelsDisplay<'a>(&'a LinkedHashSet<String>);

impl Display for LabelsDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        for label in self.0 {
            write!(f, "{label}: ")?;
        }
        Ok(())
    }
}

#[test]
fn test_format() {
    let mut node = crate::BinaryNode::default();
    // Test a couple tricky cases for `guess_c_strings()`.
    node.set_property("azbz", 0x61006200_u32.to_be_bytes().into());
    node.set_property("zabz", 0x00616200_u32.to_be_bytes().into());
    assert_eq!(
        node.get_property("azbz").unwrap().fmt_opt().unwrap(),
        "\"a\", \"b\""
    );
    assert_eq!(
        node.get_property("zabz").unwrap().fmt_opt().unwrap(),
        "< 0x616200>"
    );

    // Verify that these round-trip through the parser and evaluator.
    node.set_property("abc", c"abc".to_owned().into());
    node.set_property("allbytes", (0..=255).chain(Some(0)).collect());
    node.set_property("printable", (b' '..=b'~').chain(Some(0)).collect());

    let source = format!("/ {node};");
    let loader = crate::fs::DummyLoader;
    let arena = crate::Arena::new();
    let dts = crate::parse::parse_typed(&source, &arena).unwrap();
    let mut scribe = crate::error::Scribe::new(true);
    let (tree, node_labels, _, _) = crate::merge::merge(dts, &mut scribe);
    let tree = crate::eval::eval(tree, node_labels, &loader, &mut scribe);
    assert!(scribe.report(&loader, &mut std::io::stderr()));
    assert!(tree.children.is_empty());
    assert_eq!(tree.properties, node.properties);
}
