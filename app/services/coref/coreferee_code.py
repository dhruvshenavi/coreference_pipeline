

import spacy
import coreferee
import re

# =========================
# 1. PIPELINE
# =========================
def build_nlp(model="en_core_web_lg"):
    nlp = spacy.load(model)
    if "coreferee" not in nlp.pipe_names:
        nlp.add_pipe("coreferee", last=True)
    return nlp

# =========================
# 2. CONSTANTS
# =========================
SPEECH_VERBS = {
    "say", "said", "tell", "told", "declare", "declared",
    "add", "added", "claim", "claimed", "announce", "announced",
    "tweet", "tweeted", "post", "posted", "state", "stated",
    "remark", "remarked", "assert", "asserted"
}

PRONOUNS = {"me", "my", "he", "him", "his", "she", "her", "they", "them", "their"}

DATE_TIME_ENTS = {"DATE", "TIME"}

# =========================
# 3. UTILITIES
# =========================
def mention_to_span(mention, doc):
    if hasattr(mention, "span"):
        return mention.span
    if hasattr(mention, "token_indexes"):
        idx = list(mention.token_indexes)
        if idx:
            return doc[idx[0]: idx[-1]+1]
    if hasattr(mention, "root"):
        return mention.root.doc[mention.root.i:mention.root.i+1]
    return None

def normalize(text):
    return re.sub(r"[^a-z]", "", text.lower())

def span_signature(span):
    return {normalize(t.text) for t in span if t.pos_ in {"PROPN", "NOUN"} and normalize(t.text)}

def is_title_like(span):
    return any(t.lemma_.lower() in {"prime", "minister", "president", "pm", "chancellor"} for t in span)

# =========================
# 4. SPEAKER AND GENERIC RULES
# =========================
def is_narrator_pronoun(token):
    """Pronoun is narrator if it is subject of a speech verb."""
    return (
        token.pos_ == "PRON"
        and token.dep_ == "nsubj"
        and token.head.pos_ == "VERB"
        and token.head.lemma_.lower() in SPEECH_VERBS
    )

def is_generic_he(token):
    """Generic 'he' bound to quantifiers like 'every/each'."""
    for anc in token.ancestors:
        if anc.lemma_.lower() in {"every", "each"}:
            return True
    return False

def nearest_person_before(token):
    """Nearest PERSON entity before token, excluding DATE/TIME."""
    best = None
    best_dist = float("inf")
    for ent in token.doc.ents:
        if ent.label_ != "PERSON":
            continue
        if ent.end > token.i:
            continue
        dist = token.i - ent.end
        if dist < best_dist:
            best = ent
            best_dist = dist
    return best

def detect_speaker(sent, last_speaker=None):
    """Detect speaker entity based on speech verbs in sentence."""
    doc = sent.doc
    for token in sent:
        if token.lemma_.lower() in SPEECH_VERBS:
            # Look for proper noun subjects of the verb
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    # Find PERSON entity that contains this token
                    for ent in doc.ents:
                        if ent.label_ == "PERSON" and child.i >= ent.start and child.i < ent.end:
                            return ent
    return last_speaker  # fallback to previous speaker


# =========================
# 5. COREf CHAIN MERGING
# =========================
def pick_representative(spans):
    """
    Pick the "best" span as representative for a chain.
    Excludes DATE/TIME entities to avoid replacing people with dates.
    Prefers PERSON entities, then longest proper noun span.
    """
    def score(span):
        # Check if span is PERSON
        is_person = any(t.ent_type_ == "PERSON" for t in span)
        propn_count = sum(t.pos_ == "PROPN" for t in span)
        length = len(span)
        return (is_person, propn_count, length)

    # Filter out spans that are purely DATE/TIME
    filtered_spans = [s for s in spans if not all(t.ent_type_ in {"DATE", "TIME"} for t in s)]

    if not filtered_spans:
        return max(spans, key=lambda s: len(s))  # fallback if all are DATE/TIME

    return max(filtered_spans, key=score)


def merge_similar_chains(doc):
    raw = []
    for chain in doc._.coref_chains:
        spans = []
        for m in chain.mentions:
            sp = mention_to_span(m, doc)
            if sp:
                spans.append(sp)
        if spans:
            raw.append(spans)

    merged = []
    used = set()
    for i, spans1 in enumerate(raw):
        if i in used:
            continue
        sig1 = set().union(*(span_signature(s) for s in spans1))
        merged_spans = spans1[:]
        for j, spans2 in enumerate(raw[i+1:], start=i+1):
            if j in used:
                continue
            sig2 = set().union(*(span_signature(s) for s in spans2))
            if sig1 & sig2:
                merged_spans.extend(spans2)
                used.add(j)
            elif any(is_title_like(s) for s in spans1):
                merged_spans.extend(spans2)
                used.add(j)
        merged.append(merged_spans)
    return merged

# =========================
# 6. TOKEN → REPLACEMENT MAP
# =========================
def build_token_map(doc):
    token_map = {}
    merged_chains = merge_similar_chains(doc)
    for spans in merged_chains:
        rep = pick_representative(spans)
        for sp in spans:
            for t in sp:
                token_map[t.i] = rep
    return token_map

# =========================
# 7. RESOLUTION ENGINE
# =========================
def resolve_sentence(sent, doc, token_map, current_speaker):
    out = []
    i = sent.start
    in_quote = False

    while i < sent.end:
        token = doc[i]

        # Skip dates/times
        if token.ent_type_ in DATE_TIME_ENTS:
            out.append(token.text + token.whitespace_)
            i += 1
            continue

        # Toggle quote tracking
        if token.text in {'"', '“', '”'}:
            in_quote = not in_quote
            out.append(token.text)
            i += 1
            continue

        

        # PRONOUN resolution
        if token.pos_ == "PRON" and token.lower_ in PRONOUNS:
            # Inside quotes → prefer current speaker
            if in_quote and current_speaker:
                out.append(current_speaker.text + token.whitespace_)
                i += 1
                continue
            # Standard mapping
            rep = token_map.get(i)
            if rep:
                out.append(rep.text + token.whitespace_)
                i += 1
                continue

        # default
        out.append(token.text + token.whitespace_)
        i += 1

    return "".join(out).strip()


# =========================
# 8. MAIN RESOLVER
# =========================
def resolve_text(text, nlp):
    doc = nlp(text)
    token_map = build_token_map(doc)
    current_speaker = None

    merged_chains = merge_similar_chains(doc)

    resolved_sentences = []

    for sent in doc.sents:
        current_speaker = detect_speaker(sent, current_speaker)
        resolved = resolve_sentence(sent, doc, token_map, current_speaker)
        resolved_sentences.append(resolved)

    resolved_text = " ".join(resolved_sentences)

    return merged_chains, resolved_text

