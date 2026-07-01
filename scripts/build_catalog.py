"""
Cleans the raw scraped SHL product catalog (Individual Test Solutions only)
into a normalized structure used by the retrieval + agent layer.
"""
import json
import re

KEY_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

SRC = "/mnt/user-data/uploads/shl_product_catalog.json"
OUT = "/home/claude/shl_agent/data/catalog.json"


def main():
    raw = open(SRC, encoding="utf-8").read()
    items = json.loads(raw, strict=False)

    cleaned = []
    seen_ids = set()
    for it in items:
        eid = it.get("entity_id")
        if not eid or eid in seen_ids:
            continue
        seen_ids.add(eid)

        name = (it.get("name") or "").strip()
        name = re.sub(r"\s+", " ", name)
        link = (it.get("link") or "").strip()
        if not name or not link:
            continue

        keys_full = it.get("keys") or []
        keys_codes = [KEY_CODE.get(k, "") for k in keys_full]
        keys_codes = [k for k in keys_codes if k]

        cleaned.append({
            "id": eid,
            "name": name,
            "url": link,
            "description": (it.get("description") or "").strip(),
            "keys_full": keys_full,
            "keys": keys_codes,
            "job_levels": it.get("job_levels") or [],
            "languages": it.get("languages") or [],
            "duration": (it.get("duration") or "").strip(),
            "remote_testing": it.get("remote") == "yes",
            "adaptive_irt": it.get("adaptive") == "yes",
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(cleaned)} items to {OUT}")


if __name__ == "__main__":
    main()
