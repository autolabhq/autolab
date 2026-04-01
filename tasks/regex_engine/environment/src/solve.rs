use std::collections::VecDeque;

#[derive(Clone)]
enum ClassItem {
    Byte(u8),
    Range(u8, u8),
}

#[derive(Clone)]
enum Node {
    Literal(u8),
    Dot,
    Class(Vec<ClassItem>),
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    Repeat(Box<Node>, Rep),
    Start,
    End,
}

#[derive(Clone, Copy)]
enum Rep {
    ZeroOrOne,
    ZeroOrMore,
    OneOrMore,
}

pub struct Summary {
    pub matches: u64,
    pub checksum: u64,
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self { bytes: s.as_bytes(), pos: 0 }
    }

    fn parse(mut self) -> Node {
        self.parse_alt()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn parse_alt(&mut self) -> Node {
        let mut parts = vec![self.parse_concat()];
        while self.peek() == Some(b'|') {
            self.bump();
            parts.push(self.parse_concat());
        }
        if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            Node::Alt(parts)
        }
    }

    fn parse_concat(&mut self) -> Node {
        let mut parts = Vec::new();
        while let Some(ch) = self.peek() {
            if ch == b')' || ch == b'|' {
                break;
            }
            parts.push(self.parse_repeat());
        }
        if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            Node::Concat(parts)
        }
    }

    fn parse_repeat(&mut self) -> Node {
        let mut atom = self.parse_atom();
        loop {
            atom = match self.peek() {
                Some(b'*') => {
                    self.bump();
                    Node::Repeat(Box::new(atom), Rep::ZeroOrMore)
                }
                Some(b'+') => {
                    self.bump();
                    Node::Repeat(Box::new(atom), Rep::OneOrMore)
                }
                Some(b'?') => {
                    self.bump();
                    Node::Repeat(Box::new(atom), Rep::ZeroOrOne)
                }
                _ => return atom,
            };
        }
    }

    fn parse_atom(&mut self) -> Node {
        match self.bump().unwrap() {
            b'.' => Node::Dot,
            b'^' => Node::Start,
            b'$' => Node::End,
            b'(' => {
                let expr = self.parse_alt();
                assert_eq!(self.bump(), Some(b')'));
                expr
            }
            b'[' => {
                let mut items = Vec::new();
                loop {
                    if self.peek() == Some(b']') {
                        self.bump();
                        break;
                    }
                    let start = match self.bump().unwrap() {
                        b'\\' => self.bump().unwrap(),
                        c => c,
                    };
                    if self.peek() == Some(b'-')
                        && self.bytes.get(self.pos + 1).copied() != Some(b']')
                    {
                        self.bump();
                        let end = match self.bump().unwrap() {
                            b'\\' => self.bump().unwrap(),
                            c => c,
                        };
                        items.push(ClassItem::Range(start, end));
                    } else {
                        items.push(ClassItem::Byte(start));
                    }
                }
                Node::Class(items)
            }
            b'\\' => Node::Literal(self.bump().unwrap()),
            c => Node::Literal(c),
        }
    }
}

fn class_contains(items: &[ClassItem], b: u8) -> bool {
    for item in items {
        match *item {
            ClassItem::Byte(x) => {
                if x == b {
                    return true;
                }
            }
            ClassItem::Range(lo, hi) => {
                if lo <= b && b <= hi {
                    return true;
                }
            }
        }
    }
    false
}

fn match_positions(node: &Node, text: &[u8], pos: usize) -> Vec<usize> {
    match node {
        Node::Literal(c) => {
            if pos < text.len() && text[pos] == *c {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Node::Dot => {
            if pos < text.len() {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Node::Class(items) => {
            if pos < text.len() && class_contains(items, text[pos]) {
                vec![pos + 1]
            } else {
                Vec::new()
            }
        }
        Node::Start => {
            if pos == 0 { vec![pos] } else { Vec::new() }
        }
        Node::End => {
            if pos == text.len() { vec![pos] } else { Vec::new() }
        }
        Node::Concat(parts) => {
            let mut states = vec![pos];
            for part in parts {
                let mut next = Vec::new();
                for p in states {
                    next.extend(match_positions(part, text, p));
                }
                if next.is_empty() {
                    return next;
                }
                next.sort_unstable();
                next.dedup();
                states = next;
            }
            states
        }
        Node::Alt(parts) => {
            let mut out = Vec::new();
            for part in parts {
                out.extend(match_positions(part, text, pos));
            }
            out.sort_unstable();
            out.dedup();
            out
        }
        Node::Repeat(inner, rep) => match rep {
            Rep::ZeroOrOne => {
                let mut out = vec![pos];
                out.extend(match_positions(inner, text, pos));
                out.sort_unstable();
                out.dedup();
                out
            }
            Rep::ZeroOrMore => repeat_positions(inner, text, pos, false),
            Rep::OneOrMore => repeat_positions(inner, text, pos, true),
        },
    }
}

fn repeat_positions(node: &Node, text: &[u8], pos: usize, require_one: bool) -> Vec<usize> {
    let mut out = if require_one { Vec::new() } else { vec![pos] };
    let mut queue = VecDeque::new();
    let mut seen = vec![false; text.len() + 1];
    queue.push_back(pos);
    seen[pos] = true;

    while let Some(p) = queue.pop_front() {
        for next in match_positions(node, text, p) {
            if next != p && !seen[next] {
                seen[next] = true;
                out.push(next);
                queue.push_back(next);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn is_match(node: &Node, text: &[u8]) -> bool {
    for start in 0..=text.len() {
        if !match_positions(node, text, start).is_empty() {
            return true;
        }
    }
    false
}

pub fn run_workload(patterns: &[String], haystacks: &[Vec<u8>]) -> Summary {
    let compiled: Vec<Node> = patterns.iter().map(|p| Parser::new(p).parse()).collect();
    let mut matches = 0u64;
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;

    for (pi, re) in compiled.iter().enumerate() {
        for (ti, text) in haystacks.iter().enumerate() {
            if is_match(re, text) {
                matches += 1;
                checksum ^= ((pi as u64) << 32) ^ ti as u64;
                checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }

    Summary { matches, checksum }
}
