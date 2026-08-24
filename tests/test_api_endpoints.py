import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import pytest
from backend.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get('/api/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["platform"] == "AGRIVISION AI"


def test_chat_endpoint(client):
    payload = {"query": "tomato leaf black what medicine"}
    res = client.post('/api/chat', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert "answer" in data
    assert "interpreted_query" in data
    assert len(data["evidence_items"]) > 0


def test_crop_recommend_endpoint(client):
    payload = {
        "nitrogen": 85,
        "phosphorus": 42,
        "potassium": 43,
        "temperature": 27.5,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0
    }
    res = client.post('/api/crop-recommend', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert "ml_prediction" in data
    assert "agronomic_evidence" in data


def test_fertilizer_recommend_endpoint(client):
    payload = {
        "crop": "Rice",
        "nitrogen": 30,
        "phosphorus": 40,
        "potassium": 35,
        "soil_type": "Clayey"
    }
    res = client.post('/api/fertilizer-recommend', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert "advisory" in data


def test_observability_endpoint(client):
    res = client.get('/api/observability?query=cotton+pest+spray')
    assert res.status_code == 200
    data = res.get_json()
    assert "trace" in data
    assert "query_transformation" in data
