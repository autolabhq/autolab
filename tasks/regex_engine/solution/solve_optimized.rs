#[derive(Clone)]
enum Atom {
    Byte(u8),
    Dot,
    Class(Box<[bool; 256]>),
    Start,
    End,
}

#[derive(Clone)]
enum Ast {
    Atom(Atom),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Rep(Box<Ast>, Rep),
}

#[derive(Clone, Copy)]
enum Rep {
    ZeroOrOne,
    ZeroOrMore,
    OneOrMore,
}

#[derive(Clone)]
enum Op {
    Byte(u8),
    Dot,
    Class(Box<[bool; 256]>),
    Split,
    Jump,
    Start,
    End,
    Match,
}

#[derive(Clone)]
struct Inst {
    op: Op,
    out1: usize,
    out2: usize,
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

    fn peek(&self) -> Option<u8> { self.bytes.get(self.pos).copied() }
    fn bump(&mut self) -> Option<u8> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }
    fn parse(mut self) -> Ast { self.parse_alt() }

    fn parse_alt(&mut self) -> Ast {
        let mut v = vec![self.parse_concat()];
        while self.peek() == Some(b'|') {
            self.bump();
            v.push(self.parse_concat());
        }
        if v.len() == 1 { v.pop().unwrap() } else { Ast::Alt(v) }
    }

    fn parse_concat(&mut self) -> Ast {
        let mut v = Vec::new();
        while let Some(ch) = self.peek() {
            if ch == b')' || ch == b'|' { break; }
            v.push(self.parse_repeat());
        }
        if v.len() == 1 { v.pop().unwrap() } else { Ast::Concat(v) }
    }

    fn parse_repeat(&mut self) -> Ast {
        let mut node = self.parse_atom();
        loop {
            node = match self.peek() {
                Some(b'*') => { self.bump(); Ast::Rep(Box::new(node), Rep::ZeroOrMore) }
                Some(b'+') => { self.bump(); Ast::Rep(Box::new(node), Rep::OneOrMore) }
                Some(b'?') => { self.bump(); Ast::Rep(Box::new(node), Rep::ZeroOrOne) }
                _ => return node,
            };
        }
    }

    fn parse_atom(&mut self) -> Ast {
        match self.bump().unwrap() {
            b'.' => Ast::Atom(Atom::Dot),
            b'^' => Ast::Atom(Atom::Start),
            b'$' => Ast::Atom(Atom::End),
            b'(' => {
                let x = self.parse_alt();
                assert_eq!(self.bump(), Some(b')'));
                x
            }
            b'[' => {
                let mut cls = Box::new([false; 256]);
                loop {
                    if self.peek() == Some(b']') {
                        self.bump();
                        break;
                    }
                    let lo = match self.bump().unwrap() {
                        b'\\' => self.bump().unwrap(),
                        c => c,
                    };
                    if self.peek() == Some(b'-')
                        && self.bytes.get(self.pos + 1).copied() != Some(b']')
                    {
                        self.bump();
                        let hi = match self.bump().unwrap() { b'\\' => self.bump().unwrap(), x => x };
                        for b in lo..=hi {
                            cls[b as usize] = true;
                        }
                    } else {
                        cls[lo as usize] = true;
                    }
                }
                Ast::Atom(Atom::Class(cls))
            }
            b'\\' => Ast::Atom(Atom::Byte(self.bump().unwrap())),
            c => Ast::Atom(Atom::Byte(c)),
        }
    }
}

struct Frag {
    start: usize,
    outs: Vec<(usize, u8)>,
}

fn patch(prog: &mut [Inst], outs: &[(usize, u8)], target: usize) {
    for &(idx, slot) in outs {
        if slot == 1 { prog[idx].out1 = target; } else { prog[idx].out2 = target; }
    }
}

fn append(a: &[(usize, u8)], b: &[(usize, u8)]) -> Vec<(usize, u8)> {
    let mut v = Vec::with_capacity(a.len() + b.len());
    v.extend_from_slice(a);
    v.extend_from_slice(b);
    v
}

fn emit(prog: &mut Vec<Inst>, op: Op, out1: usize, out2: usize) -> usize {
    let idx = prog.len();
    prog.push(Inst { op, out1, out2 });
    idx
}

fn compile(ast: &Ast, prog: &mut Vec<Inst>) -> Frag {
    match ast {
        Ast::Atom(atom) => match atom {
            Atom::Byte(b) => {
                let i = emit(prog, Op::Byte(*b), usize::MAX, usize::MAX);
                Frag { start: i, outs: vec![(i, 1)] }
            }
            Atom::Dot => {
                let i = emit(prog, Op::Dot, usize::MAX, usize::MAX);
                Frag { start: i, outs: vec![(i, 1)] }
            }
            Atom::Class(c) => {
                let i = emit(prog, Op::Class(c.clone()), usize::MAX, usize::MAX);
                Frag { start: i, outs: vec![(i, 1)] }
            }
            Atom::Start => {
                let i = emit(prog, Op::Start, usize::MAX, usize::MAX);
                Frag { start: i, outs: vec![(i, 1)] }
            }
            Atom::End => {
                let i = emit(prog, Op::End, usize::MAX, usize::MAX);
                Frag { start: i, outs: vec![(i, 1)] }
            }
        },
        Ast::Concat(parts) => {
            if parts.is_empty() {
                let i = emit(prog, Op::Jump, usize::MAX, usize::MAX);
                return Frag { start: i, outs: vec![(i, 1)] };
            }
            let mut frag = compile(&parts[0], prog);
            for p in &parts[1..] {
                let next = compile(p, prog);
                patch(prog, &frag.outs, next.start);
                frag = Frag { start: frag.start, outs: next.outs };
            }
            frag
        }
        Ast::Alt(parts) => {
            let mut frag = compile(parts.last().unwrap(), prog);
            for part in parts[..parts.len() - 1].iter().rev() {
                let left = compile(part, prog);
                let i = emit(prog, Op::Split, left.start, frag.start);
                frag = Frag { start: i, outs: append(&left.outs, &frag.outs) };
            }
            frag
        }
        Ast::Rep(inner, rep) => {
            let frag = compile(inner, prog);
            match rep {
                Rep::ZeroOrMore => {
                    let i = emit(prog, Op::Split, frag.start, usize::MAX);
                    patch(prog, &frag.outs, i);
                    Frag { start: i, outs: vec![(i, 2)] }
                }
                Rep::OneOrMore => {
                    let i = emit(prog, Op::Split, frag.start, usize::MAX);
                    patch(prog, &frag.outs, i);
                    Frag { start: frag.start, outs: vec![(i, 2)] }
                }
                Rep::ZeroOrOne => {
                    let i = emit(prog, Op::Split, frag.start, usize::MAX);
                    Frag { start: i, outs: append(&frag.outs, &[(i, 2)]) }
                }
            }
        }
    }
}

struct Prog {
    insts: Vec<Inst>,
    start: usize,
    anchored: bool,
}

fn compile_prog(pat: &str) -> Prog {
    let ast = Parser::new(pat).parse();
    let mut prog = Vec::new();
    let frag = compile(&ast, &mut prog);
    let m = emit(&mut prog, Op::Match, usize::MAX, usize::MAX);
    patch(&mut prog, &frag.outs, m);
    let anchored = pat.as_bytes().first() == Some(&b'^');
    Prog { insts: prog, start: frag.start, anchored }
}

fn add_state(prog: &Prog, set: &mut Vec<usize>, seen: &mut [bool], pc: usize, pos: usize, len: usize) {
    if seen[pc] { return; }
    seen[pc] = true;
    match &prog.insts[pc].op {
        Op::Split => {
            add_state(prog, set, seen, prog.insts[pc].out1, pos, len);
            add_state(prog, set, seen, prog.insts[pc].out2, pos, len);
        }
        Op::Jump => add_state(prog, set, seen, prog.insts[pc].out1, pos, len),
        Op::Start => {
            if pos == 0 { add_state(prog, set, seen, prog.insts[pc].out1, pos, len); }
        }
        Op::End => {
            if pos == len { add_state(prog, set, seen, prog.insts[pc].out1, pos, len); }
        }
        _ => set.push(pc),
    }
}

fn is_match(prog: &Prog, text: &[u8]) -> bool {
    let n = prog.insts.len();
    let mut curr = Vec::with_capacity(n);
    let mut next = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    add_state(prog, &mut curr, &mut seen, prog.start, 0, text.len());
    if curr.iter().any(|&pc| matches!(prog.insts[pc].op, Op::Match)) { return true; }

    for (i, &b) in text.iter().enumerate() {
        next.clear();
        seen.fill(false);
        for &pc in &curr {
            match &prog.insts[pc].op {
                Op::Byte(x) if *x == b => add_state(prog, &mut next, &mut seen, prog.insts[pc].out1, i + 1, text.len()),
                Op::Dot => add_state(prog, &mut next, &mut seen, prog.insts[pc].out1, i + 1, text.len()),
                Op::Class(cls) if cls[b as usize] => add_state(prog, &mut next, &mut seen, prog.insts[pc].out1, i + 1, text.len()),
                _ => {}
            }
        }
        if !prog.anchored {
            add_state(prog, &mut next, &mut seen, prog.start, i + 1, text.len());
        }
        if next.iter().any(|&pc| matches!(prog.insts[pc].op, Op::Match)) { return true; }
        std::mem::swap(&mut curr, &mut next);
    }

    seen.fill(false);
    next.clear();
    for &pc in &curr {
        add_state(prog, &mut next, &mut seen, pc, text.len(), text.len());
    }
    next.iter().any(|&pc| matches!(prog.insts[pc].op, Op::Match))
}

pub fn run_workload(patterns: &[String], haystacks: &[Vec<u8>]) -> Summary {
    let progs: Vec<Prog> = patterns.iter().map(|p| compile_prog(p)).collect();
    let mut matches = 0u64;
    let mut checksum = 0xcbf2_9ce4_8422_2325u64;
    for (pi, prog) in progs.iter().enumerate() {
        for (ti, text) in haystacks.iter().enumerate() {
            if is_match(prog, text) {
                matches += 1;
                checksum ^= ((pi as u64) << 32) ^ ti as u64;
                checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    Summary { matches, checksum }
}
