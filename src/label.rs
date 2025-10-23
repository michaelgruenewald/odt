use crate::error::SourceError;
use crate::node::Node;
use crate::parse::TypedRuleExt;
use crate::parse::rules::{NodeReference, PropertyReference};
use crate::path::NodePath;
use hashlink::LinkedHashMap;

pub type LabelMap = LinkedHashMap<String, NodePath>;

pub struct LabelResolver<'a, P>(pub &'a LabelMap, pub &'a Node<P>);

impl<P> LabelResolver<'_, P> {
    pub fn resolve(
        &self,
        relative_to: &NodePath,
        noderef: &NodeReference,
    ) -> Result<NodePath, SourceError> {
        self.resolve_str(relative_to, noderef.str())
            .ok_or_else(|| noderef.err("no such node"))
    }

    pub fn resolve_str(&self, relative_to: &NodePath, noderef: &str) -> Option<NodePath> {
        let path = noderef.trim_matches(['&', '$', '{', '}']);
        let mut segments = path.split('/');
        let first = segments.next().unwrap();
        let segments = segments.filter(|s| !s.is_empty());
        let (root, mut result) = match first {
            "" => {
                // The node reference is absolute.
                (self.1, NodePath::root())
            }
            "." => {
                // The node reference is relative to the current node.
                let current: &Node<P> = self.1.walk(relative_to.segments()).unwrap();
                (current, relative_to.clone())
            }
            label => {
                // The first segment is a label name.
                let target = self.0.get(label)?;
                (self.1.walk(target.segments())?, target.clone())
            }
        };
        // Check that the path exists.
        root.walk(segments.clone())?;
        for s in segments {
            result.push(s);
        }
        Some(result)
    }

    pub(crate) fn prop_from_prop_ref(
        &self,
        relative_to: &NodePath,
        propref: &PropertyReference,
    ) -> Result<(NodePath, &P), SourceError> {
        // root.walk() expects all segments to be Node elements, so strip off
        // the property name after the last '/'.
        let noderef = propref.str().rsplit_once('/').map(|(a, _)| a).unwrap_or(".");
        let nodepath = self
            .resolve_str(relative_to, noderef)
            .ok_or_else(|| propref.err("no such node"))?;

        // Lookup the property in the node
        let propname = propref
            .str()
            .rsplit_once('/')
            .map(|(_, b)| b)
            .unwrap_or(propref.str())
            .trim_matches(['$', '{', '}']);
        let prop = self
            .1
            .walk(nodepath.segments())
            .ok_or_else(|| propref.err("no such node"))?
            .get_property(propname)
            .ok_or_else(|| propref.err("no such property"))?;

        Ok((nodepath, prop))
    }
}
