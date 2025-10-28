//! Facilities for parsing DTS files and expanding "/include/" directives.

use crate::Arena;
use crate::error::{Scribe, SourceError};
use crate::fs::Loader;
use bumpalo::collections::Vec;
use core::ops::Range;
use pest::iterators::Pair;
use pest::{Parser, Span};
use rules::{Dts, DtsFile, QuotedString, TopDef};
use std::path::Path;

#[derive(pest_derive::Parser, pestle::TypedRules)]
#[grammar = "src/dts.pest"]
#[typed_mod = "rules"]
pub struct DtsParser;

/// This is pest's untyped grammar, which is convenient for clients that
/// wish to visit every character of the input.
pub type Parsed<'a> = Pair<'a, Rule>;

pub fn parse_untyped(source: &str) -> Result<Parsed<'_>, SourceError> {
    let mut it = DtsParser::parse(Rule::DtsFile, source)?;
    let dtsfile = it.next().unwrap();
    assert_eq!(dtsfile.as_rule(), Rule::DtsFile);
    assert_eq!(it.next(), None);
    Ok(dtsfile)
}

pub fn parse_typed<'i>(source: &'i str, arena: &'i Arena) -> Result<&'i Dts<'i>, SourceError> {
    let tree = parse_untyped(source)?;
    let dtsfile = DtsFile::build(tree, arena);
    Ok(dtsfile.dts)
}

pub(crate) fn parse_quoted_string<'i>(
    source: &'i str,
    arena: &'i Arena,
) -> Result<&'i QuotedString<'i>, SourceError> {
    let mut it = DtsParser::parse(Rule::QuotedString, source)?;
    let tree = it.next().unwrap();
    assert_eq!(tree.as_rule(), Rule::QuotedString);
    assert_eq!(it.next(), None);
    Ok(QuotedString::build(tree, arena))
}

/// Parse the source file named by `path`.
pub fn parse_with_includes<'a>(
    loader: &'a impl Loader,
    arena: &'a Arena,
    path: &Path,
    scribe: &mut Scribe,
) -> Dts<'a> {
    parse_concat_with_includes(loader, arena, &[path], scribe)
}

/// Parse the concatenation of the source files named by `paths`.
pub fn parse_concat_with_includes<'a>(
    loader: &'a impl Loader,
    arena: &'a Arena,
    paths: &[&Path],
    scribe: &mut Scribe,
) -> Dts<'a> {
    let mut span = Span::new("", 0, 0).unwrap();
    let mut top_def = Vec::new_in(arena);
    for path in paths {
        match loader.read_utf8(path.into()) {
            Ok(Some((_, src))) => match parse_typed(src, arena) {
                Ok(dts) => {
                    if span.as_str().is_empty() {
                        span = dts._span;
                    }
                    visit_includes(1, loader, arena, path, dts, &mut top_def, scribe);
                }
                // TODO:  is with_path() needed here?
                Err(e) => scribe.err(e.with_path(path)),
            },
            _ => {
                // TODO:  presumably there is some kind of filesystem error we could propagate
                scribe.err(SourceError::new_unattributed(format!(
                    "can't load file {path:?}"
                )));
            }
        }
    }
    Dts {
        _span: span,
        top_def: arena.alloc(top_def),
    }
}

fn visit_includes<'a>(
    depth: usize,
    loader: &'a impl Loader,
    arena: &'a Arena,
    path: &Path,
    dts: &Dts<'a>,
    out: &mut Vec<&'a rules::TopDef<'a>>,
    scribe: &mut Scribe,
) {
    let dir = path.parent().unwrap();
    for top_def in dts.top_def {
        out.push(top_def);
        let TopDef::Include(include) = top_def else {
            continue;
        };
        // `dtc` has a different limit: it checks that no more than 200 files are parsed,
        // regardless of nesting.
        if depth >= 100 {
            scribe.err(include.err("includes nested too deeply"));
            return;
        }
        let pathspan = include.quoted_string.trim_one();
        // The path is not unescaped in any way before use.
        match loader.find_utf8(dir, Path::new(pathspan.as_str())) {
            Ok(Some((ipath, src))) => match parse_typed(src, arena) {
                Ok(dts) => visit_includes(depth + 1, loader, arena, ipath, dts, out, scribe),
                Err(e) => scribe.err(e.with_path(ipath)),
            },
            // TODO:  distinguish UTF-8 errors here (Err(...) vs Ok(None))
            _ => scribe.err(pathspan.err("can't find include file on search path")),
        }
    }
}

// Implements the unstable method `str::substr_range()`.
fn substr_range(outer: &str, inner: &str) -> Option<Range<usize>> {
    let outer = outer.as_bytes().as_ptr_range();
    let outer = outer.start as usize..outer.end as usize;
    let inner = inner.as_bytes().as_ptr_range();
    let inner = inner.start as usize..inner.end as usize;
    if outer.start <= inner.start && outer.end >= inner.end {
        Some(inner.start - outer.start..inner.end - outer.start)
    } else {
        None
    }
}

pub trait SpanExt {
    // TODO: Into<String> here is error-prone, because it allows forgetting `format!()`.
    fn err(&self, message: impl Into<String>) -> SourceError {
        SourceError::new(message.into(), self.span())
    }
    fn err_at(&self, substr: &str, message: impl Into<String>) -> SourceError {
        let span = self.span();
        let range = substr_range(span.as_str(), substr);
        let span = range.map(|r| span.get(r).unwrap()).unwrap_or(span);
        SourceError::new(message.into(), span)
    }
    fn span(&self) -> Span<'_>;
}

impl SpanExt for Span<'_> {
    fn span(&self) -> Span<'_> {
        *self
    }
}

pub trait TypedRuleExt<'a> {
    fn err(&self, message: impl Into<String>) -> SourceError;
    fn str(&self) -> &'a str;
    fn trim_one(&self) -> Span<'a>;
}

impl<'a, T: rules::TypedRule<'a>> TypedRuleExt<'a> for T {
    fn err(&self, message: impl Into<String>) -> SourceError {
        self.span().err(message.into())
    }
    fn str(&self) -> &'a str {
        self.span().as_str()
    }
    fn trim_one(&self) -> Span<'a> {
        let span = self.span();
        let n = span.as_str().len();
        assert!(n >= 2, "{}", self.err("no end chars to trim"));
        span.get(1..n - 1).unwrap()
    }
}
