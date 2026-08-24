"""
AGRIVISION AI - Pest Intelligence & Integrated Pest Management (IPM) Module
"""

from typing import Dict, Any, List, Optional
import re

PEST_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    'aphids': {
        'common_name': 'Aphids (Plant Lice)',
        'scientific_name': 'Aphis gossypii / Myzus persicae',
        'crops_affected': ['Cotton', 'Tomato', 'Potato', 'Mustard', 'Chili', 'Vegetables'],
        'symptoms': 'Colonies on young shoots and leaf undersides, curled leaves, sticky honeydew secretion, black sooty mold growth.',
        'lifecycle': 'Rapid reproduction with multiple generations per season; winged forms migrate between crops.',
        'ipm_cultural': 'Install yellow sticky traps (15-20 traps/acre); spray strong water jets to dislodge colonies.',
        'ipm_biological': 'Release ladybird beetles (Coccinella septempunctata), chrysoperla lacewings, or apply Verticillium lecanii fungus.',
        'chemical_safeguard': 'Apply neem oil (1500 ppm @ 5 ml/L) or registered systemic insecticide (e.g. imidacloprid / thiamethoxam) only if threshold exceeds 20% infested shoots.'
    },
    'whitefly': {
        'common_name': 'Whitefly',
        'scientific_name': 'Bemisia tabaci',
        'crops_affected': ['Cotton', 'Tomato', 'Tobacco', 'Soybean', 'Cucurbits'],
        'symptoms': 'Tiny white moth-like insects under leaves, yellow chlorotic mosaic on foliage, vector for Tomato Yellow Leaf Curl Virus (TYLCV) and Cotton Leaf Curl Virus.',
        'lifecycle': 'Egg to adult in 15-25 days; thrives in hot, dry conditions.',
        'ipm_cultural': 'Use reflective silver mulches, yellow sticky traps, avoid excess nitrogenous fertilization.',
        'ipm_biological': 'Encarsia formosa parasitic wasps, Beauveria bassiana bio-pesticide.',
        'chemical_safeguard': 'Spray insecticidal soap, neem oil, or pyriproxyfen/diafenthiuron as per registered regional label.'
    },
    'stem borer': {
        'common_name': 'Stem Borer (Yellow Stem Borer / Spotted Stem Borer)',
        'scientific_name': 'Scirpophaga incertulas / Chilo partellus',
        'crops_affected': ['Rice', 'Maize', 'Sorghum', 'Sugarcane'],
        'symptoms': '"Dead heart" in vegetative stage (central tiller dries up), "White earhead" in reproductive stage (chaffy white panicles).',
        'lifecycle': 'Moths lay egg masses covered with brown hairs on leaf tips; larvae bore into stem pith.',
        'ipm_cultural': 'Clip seedling leaf tips before transplanting; install pheromone traps (8-10 traps/acre); synchronize planting.',
        'ipm_biological': 'Release egg parasitoid Trichogramma japonicum @ 50,000/acre weekly.',
        'chemical_safeguard': 'Apply cartap hydrochloride 4G or chlorantraniliprole 0.4G granules at root zone when ETL exceeds 1 egg mass/sq.m.'
    },
    'fall armyworm': {
        'common_name': 'Fall Armyworm',
        'scientific_name': 'Spodoptera frugiperda',
        'crops_affected': ['Maize', 'Sorghum', 'Sweet Corn', 'Millets'],
        'symptoms': 'Ragged leaf feeding, window-paning, heavy sawdust-like frass inside the central whorl, destroyed tassel and cob.',
        'lifecycle': 'Voracious night feeder; inverted Y-mark on head capsule and four square spots on the 8th abdominal segment.',
        'ipm_cultural': 'Handpick egg masses and early instars; apply dry sand/sawdust mixed with lime into central whorls.',
        'ipm_biological': 'Apply Bacillus thuringiensis (Bt @ 2g/L) or Nomuraea rileyi / Spodoptera NPV.',
        'chemical_safeguard': 'Whorl application of emamectin benzoate 5% SG (@ 0.4g/L) or spinetoram 11.7% SC under strict label guidance.'
    },
    'bollworm': {
        'common_name': 'Cotton Bollworm (American Bollworm / Pink Bollworm)',
        'scientific_name': 'Helicoverpa armigera / Pectinophora gossypiella',
        'crops_affected': ['Cotton', 'Tomato', 'Chickpea', 'Pigeonpea'],
        'symptoms': 'Circular bore holes in squares and bolls, rosette flowers, prematurely opened damaged bolls.',
        'lifecycle': 'Moth lays spherical yellowish eggs singly on leaves/buds; caterpillars feed internally.',
        'ipm_cultural': 'Grow castor or marigold as trap crops; install delta pheromone traps; deep summer ploughing.',
        'ipm_biological': 'Spray HaNPV (250 LE/acre) or release Trichogramma chilonis.',
        'chemical_safeguard': 'Apply indoxacarb 14.5% SC or flubendiamide 39.35% SC strictly as per label directions.'
    },
    'thrips': {
        'common_name': 'Thrips (Chilli Thrips / Western Flower Thrips)',
        'scientific_name': 'Scirtothrips dorsalis / Frankliniella occidentalis',
        'crops_affected': ['Chili', 'Capsicum', 'Onion', 'Cotton', 'Groundnut'],
        'symptoms': 'Upward curling of leaves ("boat-shaped"), bronze raspy streaks under leaves, flower shedding, vector for TSWV virus.',
        'lifecycle': 'Microscopic slender insects with fringed wings; rapid life cycle (14-20 days).',
        'ipm_cultural': 'Install blue and yellow sticky traps (25 traps/acre); intercrop with barrier crops (maize/sorghum).',
        'ipm_biological': 'Predatory mites (Amblyseius swirskii), application of Lecanicillium lecanii.',
        'chemical_safeguard': 'Foliar spray with spinosad 45% SC (@ 0.3ml/L) or fipronil 5% SC at early pest threshold.'
    },
    'red spider mite': {
        'common_name': 'Red Spider Mite / Two-Spotted Spider Mite',
        'scientific_name': 'Tetranychus urticae',
        'crops_affected': ['Tomato', 'Brinjal', 'Cotton', 'Tea', 'Cucurbits'],
        'symptoms': 'Stippling and yellow speckling on upper leaf surfaces, dense web under leaf, bronzing and premature leaf drop.',
        'lifecycle': 'Microscopic red/yellow mites prolific during hot, dry, low-humidity spells.',
        'ipm_cultural': 'Frequent overhead sprinkler irrigation to wash webs; destroy weed hosts around field borders.',
        'ipm_biological': 'Release predatory phytoseiid mites or apply neem formulations (Azadirachtin 1%).',
        'chemical_safeguard': 'Spray wettable sulfur 80% WP (@ 3g/L) or propargite 57% EC / spiromesifen 22.9% SC.'
    }
}


class PestDetector:
    """Intelligent Pest Diagnosis and IPM Advisory Service."""

    def __init__(self):
        self.kb = PEST_KNOWLEDGE_BASE

    def diagnose_from_text(self, description_or_name: str) -> Dict[str, Any]:
        """Identifies target pests from symptom description or common names."""
        query = description_or_name.lower().strip()
        matched_key = None

        for key in self.kb.keys():
            if key in query or any(w in query for w in key.split()):
                matched_key = key
                break

        # Check symptom matching
        if not matched_key:
            for key, data in self.kb.items():
                if any(symptom_term in query for symptom_term in ['curling', 'yellow', 'hole', 'bore', 'web', 'sticky', 'caterpillar', 'dead heart']):
                    matched_key = key
                    break

        pest_data = self.kb.get(matched_key or 'aphids')
        return {
            'identified_pest': pest_data['common_name'],
            'scientific_name': pest_data['scientific_name'],
            'crops_affected': pest_data['crops_affected'],
            'symptoms': pest_data['symptoms'],
            'lifecycle': pest_data['lifecycle'],
            'ipm_management': {
                'cultural_control': pest_data['ipm_cultural'],
                'biological_control': pest_data['ipm_biological'],
                'chemical_safeguards': pest_data['chemical_safeguard']
            },
            'confidence': 0.91 if matched_key else 0.72
        }


# Global Singleton
pest_detector = PestDetector()
