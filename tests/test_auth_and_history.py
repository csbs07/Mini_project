"""Comprehensive integration tests for authentication, route protection, and isolated chat history."""
import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.server import app
from app.db import init_db, get_db_connection


class TestAuthAndChatHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize DB
        init_db()
        cls.client = TestClient(app, follow_redirects=False)

    def setUp(self):
        # Clean users/sessions/conversations for clean test runs
        with get_db_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
            conn.execute("DELETE FROM user_sessions")
            conn.execute("DELETE FROM users")
            conn.commit()

    def test_unauthenticated_redirection_and_protection(self):
        """Unauthenticated requests to /chat redirect to /login; API routes return 401."""
        # Page redirect
        res = self.client.get("/chat")
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/login")

        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 303)
        self.assertEqual(res_root.headers.get("location"), "/login")

        # Protected API endpoints
        res_me = self.client.get("/api/auth/me")
        self.assertEqual(res_me.status_code, 401)

        res_convs = self.client.get("/api/conversations")
        self.assertEqual(res_convs.status_code, 401)

        res_search = self.client.post("/api/search", json={"query": "diffusion models", "session_id": "test_s"})
        self.assertEqual(res_search.status_code, 401)

    def test_signup_validation_and_duplicate_prevention(self):
        """Input validation enforces password length, username formatting, and uniqueness."""
        # Short password
        r = self.client.post("/api/auth/signup", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "short",
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("at least 8 characters", r.json()["detail"])

        # Invalid username characters
        r = self.client.post("/api/auth/signup", json={
            "username": "alice with spaces",
            "email": "alice@example.com",
            "password": "Password123!",
        })
        self.assertEqual(r.status_code, 400)

        # Successful registration
        r_ok = self.client.post("/api/auth/signup", json={
            "username": "alice",
            "email": "alice@example.com",
            "password": "Password123!",
        })
        self.assertEqual(r_ok.status_code, 200)
        data = r_ok.json()
        self.assertEqual(data["user"]["username"], "alice")
        self.assertIn("litsearch_session", r_ok.cookies)

        # Duplicate username
        r_dup_user = self.client.post("/api/auth/signup", json={
            "username": "alice",
            "email": "different_alice@example.com",
            "password": "Password123!",
        })
        self.assertEqual(r_dup_user.status_code, 400)
        self.assertIn("already exists", r_dup_user.json()["detail"])

        # Duplicate email
        r_dup_email = self.client.post("/api/auth/signup", json={
            "username": "alice2",
            "email": "alice@example.com",
            "password": "Password123!",
        })
        self.assertEqual(r_dup_email.status_code, 400)
        self.assertIn("already exists", r_dup_email.json()["detail"])

    def test_login_and_wrong_password(self):
        """Login validates passwords securely with bcrypt and handles invalid credentials."""
        # Create user
        self.client.post("/api/auth/signup", json={
            "username": "carol",
            "email": "carol@example.com",
            "password": "ValidPassword123!",
        })

        client2 = TestClient(app, follow_redirects=False)

        # Wrong password
        r_wrong = client2.post("/api/auth/login", json={
            "identifier": "carol",
            "password": "WrongPassword123!",
        })
        self.assertEqual(r_wrong.status_code, 401)
        self.assertIn("Invalid", r_wrong.json()["detail"])

        # Unknown user
        r_unknown = client2.post("/api/auth/login", json={
            "identifier": "nonexistent_user",
            "password": "ValidPassword123!",
        })
        self.assertEqual(r_unknown.status_code, 401)

        # Correct password via username
        r_login_user = client2.post("/api/auth/login", json={
            "identifier": "carol",
            "password": "ValidPassword123!",
        })
        self.assertEqual(r_login_user.status_code, 200)
        self.assertIn("litsearch_session", r_login_user.cookies)

        # Correct password via email
        client3 = TestClient(app, follow_redirects=False)
        r_login_email = client3.post("/api/auth/login", json={
            "identifier": "carol@example.com",
            "password": "ValidPassword123!",
        })
        self.assertEqual(r_login_email.status_code, 200)
        self.assertIn("litsearch_session", r_login_email.cookies)

    def test_logout(self):
        """Logging out revokes session and clears cookie."""
        client = TestClient(app, follow_redirects=False)
        client.post("/api/auth/signup", json={
            "username": "dave",
            "email": "dave@example.com",
            "password": "Password123!",
        })

        # Can access /api/auth/me
        r_me = client.get("/api/auth/me")
        self.assertEqual(r_me.status_code, 200)

        # Log out
        r_logout = client.post("/api/auth/logout")
        self.assertEqual(r_logout.status_code, 200)

        # Now accessing /api/auth/me returns 401
        r_me_after = client.get("/api/auth/me")
        self.assertEqual(r_me_after.status_code, 401)

    def test_chat_persistence_auto_title_and_resumption(self):
        """Streaming chat turns are persisted to SQLite and can be reloaded across sessions."""
        client = TestClient(app, follow_redirects=False)
        client.post("/api/auth/signup", json={
            "username": "eve",
            "email": "eve@example.com",
            "password": "Password123!",
        })

        conv_id = "conv_test_123"
        prompt = "Explain graph neural networks for drug discovery"

        # Stream a chat turn
        res = client.post("/api/chat/stream", json={"session_id": conv_id, "message": prompt})
        self.assertEqual(res.status_code, 200)
        events = res.text
        self.assertIn('"type": "done"', events)

        # Check conversations list
        r_convs = client.get("/api/conversations")
        self.assertEqual(r_convs.status_code, 200)
        convs = r_convs.json()["conversations"]
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0]["id"], conv_id)
        # Verify auto-generated title derived from first prompt
        self.assertIn("Explain graph neural networks", convs[0]["title"])

        # Check messages loaded for conversation
        r_msgs = client.get(f"/api/conversations/{conv_id}/messages")
        self.assertEqual(r_msgs.status_code, 200)
        msgs = r_msgs.json()["messages"]
        self.assertGreaterEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], prompt)
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertTrue(len(msgs[1]["content"]) > 0)

        # Rename conversation
        r_rename = client.patch(f"/api/conversations/{conv_id}", json={"title": "GNN Drug Discovery Research"})
        self.assertEqual(r_rename.status_code, 200)

        r_convs_renamed = client.get("/api/conversations")
        self.assertEqual(r_convs_renamed.json()["conversations"][0]["title"], "GNN Drug Discovery Research")

        # Delete conversation
        r_del = client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(r_del.status_code, 200)

        r_convs_after = client.get("/api/conversations")
        self.assertEqual(len(r_convs_after.json()["conversations"]), 0)

    def test_multi_user_isolation(self):
        """User B cannot view, load, rename, or delete User A's conversations."""
        # User A
        client_a = TestClient(app, follow_redirects=False)
        client_a.post("/api/auth/signup", json={
            "username": "alice_private",
            "email": "alice_p@example.com",
            "password": "Password123!",
        })
        conv_a_id = "conv_alice_private_1"
        client_a.post("/api/chat/stream", json={
            "session_id": conv_a_id,
            "message": "Alice's proprietary secret question",
        })

        # User B
        client_b = TestClient(app, follow_redirects=False)
        client_b.post("/api/auth/signup", json={
            "username": "bob_stranger",
            "email": "bob_s@example.com",
            "password": "Password123!",
        })

        # Bob's conversation list must be empty
        r_b_convs = client_b.get("/api/conversations")
        self.assertEqual(r_b_convs.status_code, 200)
        self.assertEqual(len(r_b_convs.json()["conversations"]), 0)

        # Bob attempts to view Alice's messages -> 404 / access denied
        r_b_view = client_b.get(f"/api/conversations/{conv_a_id}/messages")
        self.assertEqual(r_b_view.status_code, 404)

        # Bob attempts to rename Alice's conversation -> 404 / access denied
        r_b_rename = client_b.patch(f"/api/conversations/{conv_a_id}", json={"title": "Hacked Title"})
        self.assertEqual(r_b_rename.status_code, 404)

        # Bob attempts to delete Alice's conversation -> 404 / access denied
        r_b_delete = client_b.delete(f"/api/conversations/{conv_a_id}")
        self.assertEqual(r_b_delete.status_code, 404)

        # Confirm Alice's conversation is still intact and unaffected
        r_a_convs = client_a.get("/api/conversations")
        self.assertEqual(len(r_a_convs.json()["conversations"]), 1)
        self.assertEqual(r_a_convs.json()["conversations"][0]["id"], conv_a_id)


if __name__ == "__main__":
    unittest.main()
