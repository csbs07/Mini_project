"""Integration test against live running server on http://127.0.0.1:8000."""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_live_workflow():
    print("Testing GET /chat...")
    r = requests.get(f"{BASE_URL}/chat")
    assert r.status_code == 200
    assert "LitSearch RAG" in r.text
    assert "Supporting Passage" in r.text
    assert "Answers use the text of retrieved open-access papers or abstracts" in r.text
    print("  -> /chat returned valid HTML with notice and drawer components.")

    sess_id = "integration_test_session"

    print("\nTesting POST /api/search...")
    search_payload = {"query": "Diffusion models for single-cell RNA sequencing", "session_id": sess_id}
    r = requests.post(f"{BASE_URL}/api/search", json=search_payload)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 0
    p1 = data["papers"][0]
    print(f"  -> Found {data['count']} papers. P1: {p1['label']} - {p1['title'][:60]}")

    print("\nTesting POST /api/papers/read...")
    read_payload = {"session_id": sess_id, "paper_id": p1["label"]}
    r = requests.post(f"{BASE_URL}/api/papers/read", json=read_payload)
    assert r.status_code == 200
    r_data = r.json()
    print(f"  -> Ingestion result: status={r_data.get('status')}, pages={r_data.get('page_count')}")

    print("\nTesting POST /api/chat/stream (SSE)...")
    chat_payload = {"session_id": sess_id, "message": "What methodology is used in P1?"}
    r = requests.post(f"{BASE_URL}/api/chat/stream", json=chat_payload, stream=True)
    assert r.status_code == 200
    
    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            ev = json.loads(line[6:])
            events.append(ev)
            if ev.get("type") == "progress":
                print(f"  [Progress Event] {ev.get('message')}")
            elif ev.get("type") == "done":
                print(f"  [Done Event] Final text snippet: {ev.get('final_text')[:100]}...")
                print(f"  [Done Event] Cost: ${ev.get('session_cost')}")
    
    assert any(e.get("type") == "done" for e in events)
    print("\nLive HTTP integration tests PASSED successfully!")

if __name__ == "__main__":
    test_live_workflow()
