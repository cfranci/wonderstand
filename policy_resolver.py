"""policy_resolver.py — decide consent tier at capture-start.

Module 2 of the Wonderstand extension. Pure stdlib at import time;
PyYAML and `phonenumbers` are *optional* runtime helpers.

Public API:
    PolicyDecision (TypedDict)
    RecordingRefused (Exception)
    resolve(counterparties, outbound, contacts, chase_state="FL") -> PolicyDecision
    load_contacts(path: Path) -> dict
    area_code_to_state(phone_e164: str) -> Optional[str]
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, TypedDict

# ---------------------------------------------------------------------------
# Two-party-consent states (per contract).
# ---------------------------------------------------------------------------
TWO_PARTY_STATES: list[str] = [
    "CA", "FL", "IL", "MA", "MD", "MT",
    "NH", "PA", "WA", "CT", "DE", "HI",
]


# ---------------------------------------------------------------------------
# Public types.
# ---------------------------------------------------------------------------
class PolicyDecision(TypedDict):
    """Decision returned by resolve(); see contract Module 2."""
    regime: Literal["one_party", "two_party", "unknown"]
    tier: Literal["A", "B", "C"]
    disclosure_required: bool
    retention_days: int
    reason: str


class RecordingRefused(Exception):
    """Raised when a counterparty has explicit consent=none."""


# ---------------------------------------------------------------------------
# NANP area-code -> state fallback (used if `phonenumbers` not installed).
# Comprehensive coverage of all 50 US states + DC.
# ---------------------------------------------------------------------------
_AREA_CODE_TO_STATE: dict[str, str] = {
    # Alabama
    "205": "AL", "251": "AL", "256": "AL", "334": "AL", "938": "AL",
    # Alaska
    "907": "AK",
    # Arizona
    "480": "AZ", "520": "AZ", "602": "AZ", "623": "AZ", "928": "AZ",
    # Arkansas
    "479": "AR", "501": "AR", "870": "AR",
    # California
    "209": "CA", "213": "CA", "279": "CA", "310": "CA", "323": "CA",
    "341": "CA", "408": "CA", "415": "CA", "424": "CA", "442": "CA",
    "510": "CA", "530": "CA", "559": "CA", "562": "CA", "619": "CA",
    "626": "CA", "628": "CA", "650": "CA", "657": "CA", "661": "CA",
    "669": "CA", "707": "CA", "714": "CA", "747": "CA", "760": "CA",
    "805": "CA", "818": "CA", "820": "CA", "831": "CA", "858": "CA",
    "909": "CA", "916": "CA", "925": "CA", "949": "CA", "951": "CA",
    # Colorado
    "303": "CO", "719": "CO", "720": "CO", "970": "CO",
    # Connecticut
    "203": "CT", "475": "CT", "860": "CT", "959": "CT",
    # Delaware
    "302": "DE",
    # DC
    "202": "DC",
    # Florida
    "239": "FL", "305": "FL", "321": "FL", "352": "FL", "386": "FL",
    "407": "FL", "561": "FL", "689": "FL", "727": "FL", "754": "FL",
    "772": "FL", "786": "FL", "813": "FL", "850": "FL", "863": "FL",
    "904": "FL", "941": "FL", "954": "FL",
    # Georgia
    "229": "GA", "404": "GA", "470": "GA", "478": "GA", "678": "GA",
    "706": "GA", "762": "GA", "770": "GA", "912": "GA",
    # Hawaii
    "808": "HI",
    # Idaho
    "208": "ID", "986": "ID",
    # Illinois
    "217": "IL", "224": "IL", "309": "IL", "312": "IL", "331": "IL",
    "618": "IL", "630": "IL", "708": "IL", "773": "IL", "779": "IL",
    "815": "IL", "847": "IL", "872": "IL",
    # Indiana
    "219": "IN", "260": "IN", "317": "IN", "463": "IN", "574": "IN",
    "765": "IN", "812": "IN", "930": "IN",
    # Iowa
    "319": "IA", "515": "IA", "563": "IA", "641": "IA", "712": "IA",
    # Kansas
    "316": "KS", "620": "KS", "785": "KS", "913": "KS",
    # Kentucky
    "270": "KY", "364": "KY", "502": "KY", "606": "KY", "859": "KY",
    # Louisiana
    "225": "LA", "318": "LA", "337": "LA", "504": "LA", "985": "LA",
    # Maine
    "207": "ME",
    # Maryland
    "240": "MD", "301": "MD", "410": "MD", "443": "MD", "667": "MD",
    # Massachusetts
    "339": "MA", "351": "MA", "413": "MA", "508": "MA", "617": "MA",
    "774": "MA", "781": "MA", "857": "MA", "978": "MA",
    # Michigan
    "231": "MI", "248": "MI", "269": "MI", "313": "MI", "517": "MI",
    "586": "MI", "616": "MI", "734": "MI", "810": "MI", "906": "MI",
    "947": "MI", "989": "MI",
    # Minnesota
    "218": "MN", "320": "MN", "507": "MN", "612": "MN", "651": "MN",
    "763": "MN", "952": "MN",
    # Mississippi
    "228": "MS", "601": "MS", "662": "MS", "769": "MS",
    # Missouri
    "314": "MO", "417": "MO", "573": "MO", "636": "MO", "660": "MO",
    "816": "MO",
    # Montana
    "406": "MT",
    # Nebraska
    "308": "NE", "402": "NE", "531": "NE",
    # Nevada
    "702": "NV", "725": "NV", "775": "NV",
    # New Hampshire
    "603": "NH",
    # New Jersey
    "201": "NJ", "551": "NJ", "609": "NJ", "640": "NJ", "732": "NJ",
    "848": "NJ", "856": "NJ", "862": "NJ", "908": "NJ", "973": "NJ",
    # New Mexico
    "505": "NM", "575": "NM",
    # New York
    "212": "NY", "315": "NY", "332": "NY", "347": "NY", "516": "NY",
    "518": "NY", "585": "NY", "607": "NY", "631": "NY", "646": "NY",
    "680": "NY", "716": "NY", "718": "NY", "838": "NY", "845": "NY",
    "914": "NY", "917": "NY", "929": "NY", "934": "NY",
    # North Carolina
    "252": "NC", "336": "NC", "704": "NC", "743": "NC", "828": "NC",
    "910": "NC", "919": "NC", "980": "NC", "984": "NC",
    # North Dakota
    "701": "ND",
    # Ohio
    "216": "OH", "220": "OH", "234": "OH", "330": "OH", "380": "OH",
    "419": "OH", "440": "OH", "513": "OH", "567": "OH", "614": "OH",
    "740": "OH", "937": "OH",
    # Oklahoma
    "405": "OK", "539": "OK", "580": "OK", "918": "OK",
    # Oregon
    "458": "OR", "503": "OR", "541": "OR", "971": "OR",
    # Pennsylvania
    "215": "PA", "223": "PA", "267": "PA", "272": "PA", "412": "PA",
    "445": "PA", "484": "PA", "570": "PA", "610": "PA", "717": "PA",
    "724": "PA", "814": "PA", "878": "PA",
    # Rhode Island
    "401": "RI",
    # South Carolina
    "803": "SC", "839": "SC", "843": "SC", "854": "SC", "864": "SC",
    # South Dakota
    "605": "SD",
    # Tennessee
    "423": "TN", "615": "TN", "629": "TN", "731": "TN", "865": "TN",
    "901": "TN", "931": "TN",
    # Texas
    "210": "TX", "214": "TX", "254": "TX", "281": "TX", "325": "TX",
    "346": "TX", "361": "TX", "409": "TX", "430": "TX", "432": "TX",
    "469": "TX", "512": "TX", "682": "TX", "713": "TX", "726": "TX",
    "737": "TX", "806": "TX", "817": "TX", "830": "TX", "832": "TX",
    "903": "TX", "915": "TX", "936": "TX", "940": "TX", "956": "TX",
    "972": "TX", "979": "TX",
    # Utah
    "385": "UT", "435": "UT", "801": "UT",
    # Vermont
    "802": "VT",
    # Virginia
    "276": "VA", "434": "VA", "540": "VA", "571": "VA", "703": "VA",
    "757": "VA", "804": "VA",
    # Washington
    "206": "WA", "253": "WA", "360": "WA", "425": "WA", "509": "WA",
    "564": "WA",
    # West Virginia
    "304": "WV", "681": "WV",
    # Wisconsin
    "262": "WI", "414": "WI", "534": "WI", "608": "WI", "715": "WI",
    "920": "WI",
    # Wyoming
    "307": "WY",
}


# ---------------------------------------------------------------------------
# Phone -> state.
# ---------------------------------------------------------------------------
def area_code_to_state(phone_e164: str) -> Optional[str]:
    """Return US state abbrev for an E.164 NANP phone, or None.

    Prefers `phonenumbers` if available; falls back to the static
    area-code dict above.
    """
    if not phone_e164 or not isinstance(phone_e164, str):
        return None
    s = phone_e164.strip()
    if not s.startswith("+1"):
        return None
    digits = "".join(ch for ch in s[2:] if ch.isdigit())
    if len(digits) < 3:
        return None
    area = digits[:3]

    # Optional: try the `phonenumbers` library for richer geocoding.
    try:
        import phonenumbers  # type: ignore
        from phonenumbers import geocoder  # type: ignore
        parsed = phonenumbers.parse(s, None)
        desc = geocoder.description_for_number(parsed, "en")
        # `desc` is usually a city/region string like "Miami, FL".
        # If it ends in a 2-letter state code, use that.
        if desc and "," in desc:
            tail = desc.rsplit(",", 1)[1].strip()
            if len(tail) == 2 and tail.isalpha():
                return tail.upper()
    except Exception:
        pass

    return _AREA_CODE_TO_STATE.get(area)


# ---------------------------------------------------------------------------
# YAML loader (PyYAML if available; tiny stdlib parser otherwise).
# ---------------------------------------------------------------------------
def load_contacts(path: Path) -> dict:
    """Parse contacts.yaml. Uses PyYAML if installed, else a minimal
    built-in parser covering exactly the schema the contract specifies.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"contacts.yaml must be a mapping, got {type(data)}")
        return data
    except ImportError:
        return _minimal_yaml_parse(text)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _minimal_yaml_parse(text: str) -> dict:
    """Tiny YAML subset:
        - top-level mappings with 2-space-indented child mappings
        - lists with `- value` items (2-space indent)
        - quoted or unquoted scalar values
        - `#` comments and blank lines ignored

    Adequate for the contacts.yaml schema in the contract. Not a general
    YAML implementation.
    """
    result: dict = {}
    cur_key: Optional[str] = None
    cur_child: Optional[dict] = None
    cur_list: Optional[list] = None

    for raw_line in text.splitlines():
        # Strip comments and trailing whitespace.
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        # Compute indent in spaces.
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)

        if indent == 0:
            # New top-level key.
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            key = _strip_quotes(key.strip())
            value = value.strip()
            if value == "":
                # Container follows.
                cur_key = key
                cur_child = {}
                cur_list = None
                result[key] = cur_child
            else:
                result[key] = _coerce_scalar(value)
                cur_key = None
                cur_child = None
                cur_list = None
        else:
            # Child line.
            if cur_key is None:
                continue
            if stripped.startswith("- "):
                # Convert current container to a list if not already.
                if cur_list is None:
                    cur_list = []
                    result[cur_key] = cur_list
                    cur_child = None
                item = _coerce_scalar(stripped[2:].strip())
                cur_list.append(item)
            else:
                if ":" not in stripped:
                    continue
                k, _, v = stripped.partition(":")
                k = _strip_quotes(k.strip())
                v = v.strip()
                if cur_child is None:
                    cur_child = {}
                    result[cur_key] = cur_child
                    cur_list = None
                cur_child[k] = _coerce_scalar(v)

    return result


def _coerce_scalar(value: str):
    v = _strip_quotes(value)
    if v == "":
        return ""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "~"):
        return None
    # Int?
    try:
        if v.lstrip("-").isdigit():
            return int(v)
    except Exception:
        pass
    return v


# ---------------------------------------------------------------------------
# Resolver.
# ---------------------------------------------------------------------------
def _tier_decision(
    tier: Literal["A", "B", "C"],
    regime: Literal["one_party", "two_party", "unknown"],
    reason: str,
) -> PolicyDecision:
    disclosure = tier == "C"
    retention = 0 if tier == "B" else 90
    return PolicyDecision(
        regime=regime,
        tier=tier,
        disclosure_required=disclosure,
        retention_days=retention,
        reason=reason,
    )


def resolve(
    counterparties: list[str],
    outbound: bool,
    contacts: dict,
    chase_state: str = "FL",
) -> PolicyDecision:
    """Return a PolicyDecision for this capture per contract Module 2 rules.

    Rules (in order):
      1. Any counterparty with consent=full      -> tier A.
      2. Any counterparty with consent=none      -> RecordingRefused.
      3. Any counterparty with consent=transcript_only -> tier B.
      4. Compute regime (Chase state or any area-code in TWO_PARTY_STATES).
      5. one_party -> A. two_party + outbound -> C. two_party + inbound -> B.
    """
    contacts = contacts or {}
    defaults = contacts.get("defaults", {}) or {}
    two_party_states = set(contacts.get("TWO_PARTY_STATES") or TWO_PARTY_STATES)

    # Per-counterparty consent lookup.
    explicit_consents: list[tuple[str, str]] = []
    for cp in counterparties or []:
        entry = contacts.get(cp)
        if isinstance(entry, dict):
            c = entry.get("consent")
            if isinstance(c, str) and c != "default":
                explicit_consents.append((cp, c))

    # Rule 1.
    for cp, c in explicit_consents:
        if c == "full":
            return _tier_decision(
                "A",
                "unknown",
                f"explicit consent=full for {cp}",
            )

    # Rule 2.
    for cp, c in explicit_consents:
        if c == "none":
            raise RecordingRefused(
                f"counterparty {cp} has consent=none; refusing to record"
            )

    # Rule 3.
    for cp, c in explicit_consents:
        if c == "transcript_only":
            return _tier_decision(
                "B",
                "unknown",
                f"explicit consent=transcript_only for {cp}",
            )

    # Rule 4: compute regime.
    states_in_play: list[str] = []
    if chase_state:
        states_in_play.append(chase_state.upper())
    for cp in counterparties or []:
        st = area_code_to_state(cp)
        if st:
            states_in_play.append(st)

    is_two_party = any(s in two_party_states for s in states_in_play)
    regime: Literal["one_party", "two_party", "unknown"] = (
        "two_party" if is_two_party else "one_party"
    )

    # Rule 5.
    if regime == "one_party":
        tier = defaults.get("unknown_one_party", "A")
        return _tier_decision(
            tier, regime,
            f"one-party regime (states={states_in_play}); default tier {tier}",
        )

    # two_party
    if outbound:
        return _tier_decision(
            "C", regime,
            f"two-party regime (states={states_in_play}), outbound -> disclose then A",
        )

    # two_party + inbound
    tier = defaults.get("unknown_two_party", "B")
    return _tier_decision(
        tier, regime,
        f"two-party regime (states={states_in_play}), inbound -> tier {tier}",
    )


# ---------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------
def _selftest() -> bool:
    contacts = {
        "+15551234567": {"name": "Maria", "consent": "full"},
        "+14155557890": {"name": "Acme Legal", "consent": "none"},
        "+12025550199": {"name": "Sam", "consent": "transcript_only"},
        "defaults": {"unknown_two_party": "B", "unknown_one_party": "A"},
        "TWO_PARTY_STATES": TWO_PARTY_STATES,
    }

    results: list[bool] = []

    # (1) explicit consent=full -> tier A.
    try:
        d = resolve(["+15551234567"], outbound=True, contacts=contacts, chase_state="FL")
        ok = d["tier"] == "A"
        print(f"[1] explicit-full -> A: {'PASS' if ok else 'FAIL'} ({d})")
        results.append(ok)
    except Exception as e:
        print(f"[1] explicit-full -> A: FAIL (unexpected {type(e).__name__}: {e})")
        results.append(False)

    # (2) explicit consent=none -> raises RecordingRefused.
    try:
        resolve(["+14155557890"], outbound=True, contacts=contacts, chase_state="FL")
        print("[2] explicit-none -> raises: FAIL (no exception)")
        results.append(False)
    except RecordingRefused as e:
        print(f"[2] explicit-none -> raises: PASS ({e})")
        results.append(True)
    except Exception as e:
        print(f"[2] explicit-none -> raises: FAIL (wrong exc {type(e).__name__}: {e})")
        results.append(False)

    # (3) two-party + outbound (no explicit consent) -> tier C.
    #     Chase in FL (two-party); unknown counterparty.
    try:
        d = resolve(["+19998887777"], outbound=True, contacts=contacts, chase_state="FL")
        ok = d["tier"] == "C" and d["regime"] == "two_party"
        print(f"[3] two-party outbound -> C: {'PASS' if ok else 'FAIL'} ({d})")
        results.append(ok)
    except Exception as e:
        print(f"[3] two-party outbound -> C: FAIL (unexpected {type(e).__name__}: {e})")
        results.append(False)

    # (4) one-party + inbound -> tier A.
    #     Chase in TX (one-party), counterparty in NY (one-party).
    try:
        d = resolve(["+12125550100"], outbound=False, contacts=contacts, chase_state="TX")
        ok = d["tier"] == "A" and d["regime"] == "one_party"
        print(f"[4] one-party inbound -> A: {'PASS' if ok else 'FAIL'} ({d})")
        results.append(ok)
    except Exception as e:
        print(f"[4] one-party inbound -> A: FAIL (unexpected {type(e).__name__}: {e})")
        results.append(False)

    return all(results)


if __name__ == "__main__":
    # Also smoke-test contacts.yaml loader if file is alongside.
    here = Path(__file__).parent
    yaml_path = here / "contacts.yaml"
    if yaml_path.exists():
        try:
            parsed = load_contacts(yaml_path)
            assert "defaults" in parsed, "defaults block missing"
            assert "TWO_PARTY_STATES" in parsed, "TWO_PARTY_STATES missing"
            print(f"[yaml] load_contacts({yaml_path.name}): PASS "
                  f"({len(parsed)} top-level keys)")
        except Exception as e:
            print(f"[yaml] load_contacts: FAIL ({type(e).__name__}: {e})")

    ok = _selftest()
    print("OK" if ok else "FAIL")
