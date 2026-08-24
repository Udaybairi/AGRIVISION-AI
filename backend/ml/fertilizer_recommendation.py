"""
AGRIVISION AI - Fertilizer Advisory & Soil Health Diagnostic Module
Computes soil NPK deficiencies, provides evidence-based nutrient management, and fertilizer schedules.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from backend.config import FERTILIZER_CSV_PATH

FERTILIZER_GUIDELINES: Dict[str, Dict[str, str]] = {
    'NHigh': {
        'status': 'Excess Nitrogen (N)',
        'symptoms': 'Excessive vegetative foliage, delayed flowering/fruiting, dark green watery leaves prone to lodging and pest attacks.',
        'action': 'Halt all nitrogenous fertilizer applications (Urea, Ammonium Nitrate). Apply light irrigation to leach soluble nitrates and top-dress with Potassium to restore balance.',
        'organic_alternative': 'Add carbon-rich organic mulch (straw, sawdust) to tie up excess available soil nitrogen.'
    },
    'Nlow': {
        'status': 'Nitrogen (N) Deficiency',
        'symptoms': 'General pale green or yellowing of older lower leaves (chlorosis), stunted root/shoot growth, early senescence.',
        'action': 'Apply Urea (46% N) at 30-50 kg/acre or Ammonium Sulfate as split doses during vegetative growth.',
        'organic_alternative': 'Apply well-decomposed Farmyard Manure (FYM), composted poultry manure, or foliar spray with Panchagavya/vermiwash.'
    },
    'PHigh': {
        'status': 'Excess Phosphorus (P)',
        'symptoms': 'Rarely toxic directly, but induces severe secondary Zinc (Zn), Iron (Fe), and Magnesium (Mg) micronutrient deficiencies.',
        'action': 'Suspend Single Superphosphate (SSP) and DAP. Apply foliar sprays of chelated Zinc (Zn-EDTA @ 1g/L) and Ferrous sulfate.',
        'organic_alternative': 'Apply biochar or humic acid to buffer excessive free phosphates in the root zone.'
    },
    'Plow': {
        'status': 'Phosphorus (P) Deficiency',
        'symptoms': 'Dark green to purplish/bronze discoloration on leaf stems and veins, poor root branching, delayed maturity.',
        'action': 'Apply Diammonium Phosphate (DAP 18-46-0) or Single Super Phosphate (SSP 16% P2O5) placed near root zone as basal dose.',
        'organic_alternative': 'Incorporate Rock Phosphate with Phosphate Solubilizing Bacteria (PSB) cultures and bone meal.'
    },
    'KHigh': {
        'status': 'Excess Potassium (K)',
        'symptoms': 'Interferes with plant absorption of Magnesium (Mg) and Calcium (Ca), causing interveinal leaf scorch.',
        'action': 'Stop Muriate of Potash (MOP) applications. Flush soil with adequate irrigation and supplement Calcium Nitrate / Epsom salt.',
        'organic_alternative': 'Incorporate gypsum (calcium sulfate) to displace potassium ions on soil exchange complexes.'
    },
    'Klow': {
        'status': 'Potassium (K) Deficiency',
        'symptoms': 'Marginal leaf scorch, yellowing/browning along leaf tips and edges (firing), weak stalks, small inferior fruits.',
        'action': 'Apply Muriate of Potash (MOP / KCl 60% K2O) or Potassium Sulfate (SOP) at 25-40 kg/acre.',
        'organic_alternative': 'Incorporate wood ash, decomposed banana peel compost, or apply potassium mobilizing bacteria (KMB).'
    }
}


class FertilizerAdvisor:
    """Production Fertilizer and Soil Nutrient Advisory Engine."""

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or FERTILIZER_CSV_PATH
        self.df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        try:
            if self.csv_path.exists():
                self.df = pd.read_csv(self.csv_path)
                self.df['Crop'] = self.df['Crop'].str.strip().str.lower()
                print(f"[FertilizerAdvisor] Loaded reference dataset with {len(self.df)} crop profiles.")
            else:
                print(f"[FertilizerAdvisor] Warning: fertilizer reference CSV not found at {self.csv_path}")
        except Exception as e:
            print(f"[FertilizerAdvisor] Error loading fertilizer data: {e}")

    def get_supported_crops(self) -> List[str]:
        if self.df is not None:
            return sorted([c.capitalize() for c in self.df['Crop'].unique()])
        return ['Rice', 'Maize', 'Cotton', 'Wheat', 'Tomato', 'Potato', 'Chickpea', 'Banana', 'Coffee']

    def recommend(
        self,
        crop_name: str,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        soil_type: str = "Loamy",
        ph: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Diagnoses nutrient imbalances against standard requirements and provides actionable advice.
        """
        clean_crop = crop_name.strip().lower()
        
        # Determine baseline standards
        if self.df is not None and clean_crop in self.df['Crop'].values:
            row = self.df[self.df['Crop'] == clean_crop].iloc[0]
            req_n, req_p, req_k = float(row['N']), float(row['P']), float(row['K'])
        else:
            # Agronomic defaults for general crops
            req_n, req_p, req_k = 80.0, 40.0, 40.0

        diff_n = req_n - nitrogen
        diff_p = req_p - phosphorus
        diff_k = req_k - potassium

        # Identify highest deviation
        deviations = {
            'N': (diff_n, 'NHigh' if diff_n < -10 else ('Nlow' if diff_n > 10 else 'N_Balanced')),
            'P': (diff_p, 'PHigh' if diff_p < -8 else ('Plow' if diff_p > 8 else 'P_Balanced')),
            'K': (diff_k, 'KHigh' if diff_k < -8 else ('Klow' if diff_k > 8 else 'K_Balanced'))
        }

        # Find primary imbalance
        abs_diffs = {'N': abs(diff_n), 'P': abs(diff_p), 'K': abs(diff_k)}
        dominant_element = max(abs_diffs, key=abs_diffs.get)
        dominant_key = deviations[dominant_element][1]

        if dominant_key in FERTILIZER_GUIDELINES:
            guideline = FERTILIZER_GUIDELINES[dominant_key]
        else:
            guideline = {
                'status': 'Optimal Nutrient Balance',
                'symptoms': 'Soil test levels are within favorable agronomic ranges for this crop.',
                'action': 'Maintain standard maintenance fertilizer schedule with balanced NPK split applications.',
                'organic_alternative': 'Continue annual application of well-rotted farmyard manure.'
            }

        # Soil pH specific advice
        ph_advice = None
        if ph is not None:
            if ph < 6.0:
                ph_advice = "Acidic soil detected (pH < 6.0): Phosphorus availability is reduced. Consider applying agricultural lime (calcium carbonate) or dolomite to raise pH."
            elif ph > 7.8:
                ph_advice = "Alkaline soil detected (pH > 7.8): Micronutrient uptake (Iron, Zinc) is restricted. Apply agricultural gypsum and elemental sulfur with organic compost."

        return {
            'crop': crop_name.capitalize(),
            'dominant_element': dominant_element,
            'status': guideline['status'],
            'symptoms': guideline['symptoms'],
            'recommended_action': guideline['action'],
            'organic_alternative': guideline['organic_alternative'],
            'soil_ph_advisory': ph_advice,
            'soil_type': soil_type,
            'nutrient_comparison': {
                'nitrogen': {'current': nitrogen, 'required': req_n, 'difference': round(diff_n, 1)},
                'phosphorus': {'current': phosphorus, 'required': req_p, 'difference': round(diff_p, 1)},
                'potassium': {'current': potassium, 'required': req_k, 'difference': round(diff_k, 1)}
            }
        }


# Global Singleton
fertilizer_advisor = FertilizerAdvisor()
