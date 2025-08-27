# enhanced_corruption_analysis.py
import json
import os
import re
from collections import Counter
import math
from datetime import datetime
from typing import Dict
import groq  # Ensure you have the groq package installed
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Enhanced Configuration
CONFIG = {
    'input_json': "data.json",
    'output_flags': "enhanced2_flags.json",
    'output_summary': "corruption2_summary.csv",
    'output_report': "corruption2_report.txt",
    'output_groq_report': "corruption2_report_public.json",
    # Set your Groq API key as environment variable
    'groq_api_key': GROQ_API_KEY,
    'groq_model': 'openai/gpt-oss-20b',
    'thresholds': {
        'sequential_run_min': 8,
        'duplicate_voter_threshold': 1,
        'age_heaping_percent': 40.0,
        'benford_pvalue': 0.01,
        'family_size_suspicious': 10,
        'age_gap_suspicious': 5,  # suspicious age gaps in families
        'name_similarity_threshold': 0.8,
        'address_clustering_eps': 0.5,
        'min_cluster_size': 5,
        'high_risk_score': 0.7,
        'medium_risk_score': 0.4
    },
    'weights': {
        'duplicate_voter_id': 0.5,
        'sequential_id': 0.4,
        'age_heaping': 0.3,
        'benford_violation': 0.35,
        'suspicious_family': 0.25,
        'photo_missing': 0.15,
        'name_pattern': 0.2,
        'address_clustering': 0.3,
        'statistical_outlier': 0.4
    }
}

"""
convert JSON to xlx file for teh personal use
fgatehr name or husband name do not change use separte variables.
"""


def load_voter_data(data: dict) -> pd.DataFrame:
    """Load and preprocess voter data directly from JSON dict (no file I/O)."""
    try:
        # Extract all voters from each page
        df = pd.DataFrame(data)

        # Standardize column names
        column_mapping = {
            'father_husband_name': 'father_name',
            'serial_number': 'serial_number',
            'name': 'name',
            'age': 'age',
            'gender': 'gender',
            'house_number': 'house_number',
            'voter_id': 'voter_id',
            'photo_status': 'photo_status'
        }
        df = df.rename(columns=column_mapping)

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def extract_numeric_id(voter_id: str) -> int:
    """Extract numeric part from voter ID"""
    if not voter_id:
        return None
    match = re.search(r'(\d+)', str(voter_id))
    return int(match.group(1)) if match else None


def clean_text_field(text: str) -> str:
    """Clean and standardize text fields"""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip().lower()

# === ENHANCED ANALYTICAL FUNCTIONS ===


def detect_voter_id_anomalies(df: pd.DataFrame) -> Dict:
    """Enhanced voter ID anomaly detection"""
    results = {
        'duplicates': [],
        'sequential_runs': [],
        'format_inconsistencies': [],
        'suspicious_patterns': []
    }

    # 1. Duplicate Voter IDs
    duplicate_counts = df['voter_id'].value_counts()
    duplicates = duplicate_counts[duplicate_counts >
                                  CONFIG['thresholds']['duplicate_voter_threshold']]

    for voter_id, count in duplicates.items():
        duplicate_records = df[df['voter_id'] == voter_id].to_dict('records')
        results['duplicates'].append({
            'voter_id': voter_id,
            'count': count,
            'records': duplicate_records
        })

    # 2. Sequential ID Runs (Enhanced)
    df['numeric_id'] = df['voter_id'].apply(extract_numeric_id)
    df_sorted = df[df['numeric_id'].notna()].sort_values('numeric_id')

    current_run = []
    sequential_runs = []

    for idx, row in df_sorted.iterrows():
        if not current_run or row['numeric_id'] == current_run[-1]['numeric_id'] + 1:
            current_run.append({
                'voter_id': row['voter_id'],
                'numeric_id': row['numeric_id'],
                'index': idx
            })
        else:
            if len(current_run) >= CONFIG['thresholds']['sequential_run_min']:
                sequential_runs.append(current_run)
            current_run = [{'voter_id': row['voter_id'],
                            'numeric_id': row['numeric_id'], 'index': idx}]

    if len(current_run) >= CONFIG['thresholds']['sequential_run_min']:
        sequential_runs.append(current_run)

    results['sequential_runs'] = sequential_runs

    # 3. Format Inconsistencies
    id_formats = df['voter_id'].apply(
        lambda x: re.sub(r'\d+', 'N', str(x)) if x else '')
    format_counts = id_formats.value_counts()
    if len(format_counts) > 5:  # Too many different formats
        results['format_inconsistencies'] = format_counts.to_dict()

    return results


def advanced_benford_analysis(df: pd.DataFrame) -> Dict:
    """Enhanced Benford's Law analysis with multiple tests"""
    results = {}

    # Test on different numeric fields
    numeric_fields = ['voter_id', 'age', 'house_number', 'serial_number']

    for field in numeric_fields:
        if field not in df.columns:
            continue

        if field == 'voter_id':
            values = df[field].apply(extract_numeric_id).dropna().astype(int)
        else:
            values = pd.to_numeric(df[field], errors='coerce').dropna()
            values = values[values > 0].astype(int)

        if len(values) < 50:  # Need sufficient data
            continue

        # First digit test
        first_digits = [int(str(v)[0]) for v in values if v > 0]
        if first_digits:
            observed = Counter(first_digits)
            observed_counts = np.array(
                [observed.get(d, 0) for d in range(1, 10)], dtype=float)
            expected_probs = np.array([math.log10(1 + 1/d)
                                      for d in range(1, 10)])
            expected_counts = expected_probs * len(first_digits)

            # Chi-square test
            chi2_stat = np.sum(
                (observed_counts - expected_counts)**2 / expected_counts)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=8)

            # Second digit test for more precision
            second_digits = []
            for v in values:
                s = str(v)
                if len(s) > 1:
                    second_digits.append(int(s[1]))

            second_digit_uniform = None
            if len(second_digits) >= 50:
                second_observed = Counter(second_digits)
                second_counts = np.array(
                    [second_observed.get(d, 0) for d in range(10)], dtype=float)
                expected_uniform = len(second_digits) / 10
                second_chi2 = np.sum(
                    (second_counts - expected_uniform)**2 / expected_uniform)
                second_p = 1 - stats.chi2.cdf(second_chi2, df=9)
                second_digit_uniform = {
                    'chi2': second_chi2, 'p_value': second_p}

            results[field] = {
                'first_digit_test': {
                    'chi2': chi2_stat,
                    'p_value': p_value,
                    'significant_deviation': p_value < CONFIG['thresholds']['benford_pvalue'],
                    'observed': dict(observed),
                    'expected': expected_counts.tolist()
                },
                'second_digit_test': second_digit_uniform,
                'sample_size': len(values)
            }

    return results


def demographic_anomaly_detection(df: pd.DataFrame) -> Dict:
    """Advanced demographic anomaly detection"""
    results = {
        'age_anomalies': {},
        'gender_anomalies': {},
        'family_anomalies': {},
        'geographic_anomalies': {}
    }

    # Age-related anomalies
    if 'age' in df.columns:
        ages = pd.to_numeric(df['age'], errors='coerce').dropna()

        # Age heaping (enhanced)
        last_digits = ages % 10
        digit_counts = last_digits.value_counts().sort_index()
        heaping_score = (digit_counts.get(0, 0) +
                         digit_counts.get(5, 0)) / len(ages) * 100

        # Age distribution analysis
        age_hist, age_bins = np.histogram(ages, bins=20)
        age_z_scores = np.abs(stats.zscore(age_hist))
        unusual_age_ranges = age_bins[:-
                                      1][age_z_scores > 2]  # Outlier detection

        # Impossible ages
        impossible_ages = ages[(ages < 18) | (ages > 120)]

        results['age_anomalies'] = {
            'heaping_percentage': heaping_score,
            'heaping_suspicious': heaping_score > CONFIG['thresholds']['age_heaping_percent'],
            'unusual_age_ranges': unusual_age_ranges.tolist(),
            'impossible_ages': impossible_ages.tolist(),
            'age_distribution': {
                'mean': float(ages.mean()),
                'std': float(ages.std()),
                'skewness': float(stats.skew(ages)),
                'kurtosis': float(stats.kurtosis(ages))
            }
        }

    # Gender ratio analysis
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        total_voters = len(df)

        # Expected ratio (approximately 52% male, 48% female in India)
        expected_male_ratio = 0.52
        if 'पुरुष' in gender_counts:
            actual_male_ratio = gender_counts['पुरुष'] / total_voters
            gender_deviation = abs(actual_male_ratio - expected_male_ratio)

            results['gender_anomalies'] = {
                'actual_male_ratio': actual_male_ratio,
                'expected_male_ratio': expected_male_ratio,
                'deviation': gender_deviation,
                'suspicious': gender_deviation > 0.15,  # 15% deviation threshold
                'counts': gender_counts.to_dict()
            }

    # Family structure anomalies
    if 'house_number' in df.columns and 'father_name' in df.columns:
        # Create family groups
        df['family_key'] = df['house_number'].astype(
            str) + '_' + df['father_name'].fillna('')
        family_sizes = df['family_key'].value_counts()

        # Analyze family age distributions
        suspicious_families = []
        for family_key, size in family_sizes.items():
            if size >= CONFIG['thresholds']['family_size_suspicious']:
                family_members = df[df['family_key'] == family_key]
                if 'age' in family_members.columns:
                    ages = pd.to_numeric(
                        family_members['age'], errors='coerce').dropna()
                    if len(ages) > 1:
                        age_range = ages.max() - ages.min()
                        # Suspicious if all family members have very similar ages
                        if age_range <= CONFIG['thresholds']['age_gap_suspicious']:
                            suspicious_families.append({
                                'family_key': family_key,
                                'size': size,
                                'age_range': age_range,
                                'ages': ages.tolist()
                            })

        results['family_anomalies'] = {
            'large_families': family_sizes[family_sizes >= CONFIG['thresholds']['family_size_suspicious']].to_dict(),
            'suspicious_age_patterns': suspicious_families,
            'average_family_size': float(family_sizes.mean())
        }

    return results


def name_pattern_analysis(df: pd.DataFrame) -> Dict:
    """Analyze name patterns for suspicious repetitions"""
    results = {
        'repeated_names': {},
        'name_length_anomalies': {},
        'suspicious_patterns': []
    }

    if 'name' not in df.columns:
        return results

    # Name frequency analysis
    names = df['name'].fillna('').str.strip()
    name_counts = names.value_counts()
    # Names appearing more than 5 times
    repeated_names = name_counts[name_counts > 5]

    # Name length analysis
    name_lengths = names.str.len()
    length_stats = {
        'mean': float(name_lengths.mean()),
        'std': float(name_lengths.std()),
        'unusual_short': len(name_lengths[name_lengths < 3]),
        'unusual_long': len(name_lengths[name_lengths > 30])
    }

    # Pattern detection (e.g., sequential naming)
    pattern_suspects = []
    for name, count in repeated_names.items():
        if count > 10:  # High repetition threshold
            records = df[df['name'] == name]
            # Check if these records are clustered together
            if 'serial_number' in records.columns:
                serials = pd.to_numeric(
                    records['serial_number'], errors='coerce').dropna()
                if len(serials) > 1:
                    serial_gaps = np.diff(sorted(serials))
                    if np.mean(serial_gaps) < 5:  # Names appear close together
                        pattern_suspects.append({
                            'name': name,
                            'count': count,
                            'avg_serial_gap': float(np.mean(serial_gaps))
                        })

    results = {
        'repeated_names': repeated_names.to_dict(),
        'name_length_anomalies': length_stats,
        'suspicious_patterns': pattern_suspects
    }

    return results


def clustering_analysis(df: pd.DataFrame) -> Dict:
    """Perform clustering analysis to detect artificial groupings"""
    results = {'address_clusters': [], 'demographic_clusters': []}

    if len(df) < 20:  # Need sufficient data for clustering
        return results

    # Address-based clustering
    if 'house_number' in df.columns:
        # Convert house numbers to numeric for clustering
        house_numbers = pd.to_numeric(
            df['house_number'], errors='coerce').fillna(0)

        if house_numbers.nunique() > 5:  # Need variety for clustering
            X = house_numbers.values.reshape(-1, 1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clustering = DBSCAN(eps=CONFIG['thresholds']['address_clustering_eps'],
                                min_samples=CONFIG['thresholds']['min_cluster_size'])
            cluster_labels = clustering.fit_predict(X_scaled)

            # Analyze clusters
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_mask = cluster_labels == label
                    cluster_size = np.sum(cluster_mask)
                    if cluster_size >= CONFIG['thresholds']['min_cluster_size']:
                        cluster_houses = house_numbers[cluster_mask].tolist()
                        results['address_clusters'].append({
                            'cluster_id': int(label),
                            'size': int(cluster_size),
                            'house_numbers': cluster_houses
                        })

    return results


def calculate_risk_scores(df: pd.DataFrame, analyses: Dict) -> pd.DataFrame:
    """Calculate comprehensive risk scores for each record"""
    df_scored = df.copy()
    df_scored['risk_score'] = 0.0
    df_scored['risk_flags'] = df_scored.index.map(lambda x: [])

    # Apply various risk factors
    for idx, row in df_scored.iterrows():
        score = 0.0
        flags = []

        # Duplicate voter ID
        voter_id = row.get('voter_id', '')
        duplicate_ids = [d['voter_id'] for d in analyses.get(
            'id_anomalies', {}).get('duplicates', [])]
        if voter_id in duplicate_ids:
            score += CONFIG['weights']['duplicate_voter_id']
            flags.append('duplicate_voter_id')

        # Sequential ID pattern
        sequential_ids = []
        for run in analyses.get('id_anomalies', {}).get('sequential_runs', []):
            sequential_ids.extend([r['voter_id'] for r in run])
        if voter_id in sequential_ids:
            score += CONFIG['weights']['sequential_id']
            flags.append('sequential_id_pattern')

        # Age heaping
        age = pd.to_numeric(row.get('age', 0), errors='coerce')
        if not pd.isna(age) and (age % 10 == 0 or age % 10 == 5):
            age_analysis = analyses.get(
                'demographic_anomalies', {}).get('age_anomalies', {})
            if age_analysis.get('heaping_suspicious', False):
                score += CONFIG['weights']['age_heaping']
                flags.append('age_heaping')

        # Family size anomaly
        if 'house_number' in row and 'father_name' in row:
            family_key = str(row['house_number']) + '_' + \
                str(row.get('father_name', ''))
            large_families = analyses.get('demographic_anomalies', {}).get(
                'family_anomalies', {}).get('large_families', {})
            if family_key in large_families:
                score += CONFIG['weights']['suspicious_family']
                flags.append('large_family')

        # Photo missing
        photo_status = str(row.get('photo_status', '')).lower()
        if 'उपलब्ध' not in photo_status or photo_status == '':
            score += CONFIG['weights']['photo_missing']
            flags.append('photo_missing')

        # Name pattern anomaly
        name = row.get('name', '')
        repeated_names = analyses.get(
            'name_patterns', {}).get('repeated_names', {})
        if name in repeated_names and repeated_names[name] > 10:
            score += CONFIG['weights']['name_pattern']
            flags.append('repeated_name_pattern')

        df_scored.at[idx, 'risk_score'] = min(1.0, score)  # Cap at 1.0
        df_scored.at[idx, 'risk_flags'] = flags

    return df_scored


def generate_comprehensive_report(df: pd.DataFrame, analyses: Dict, df_scored: pd.DataFrame) -> str:
    """Generate a comprehensive corruption analysis report"""
    report = []
    report.append("="*80)
    report.append("ELECTION DATA CORRUPTION ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Records Analyzed: {len(df):,}")
    report.append("")

    # Risk Summary
    high_risk = len(df_scored[df_scored['risk_score']
                    >= CONFIG['thresholds']['high_risk_score']])
    medium_risk = len(df_scored[(df_scored['risk_score'] >= CONFIG['thresholds']['medium_risk_score']) &
                                (df_scored['risk_score'] < CONFIG['thresholds']['high_risk_score'])])
    low_risk = len(df_scored) - high_risk - medium_risk

    report.append("RISK DISTRIBUTION:")
    report.append(
        f"  High Risk (≥{CONFIG['thresholds']['high_risk_score']}): {high_risk:,} ({high_risk/len(df)*100:.1f}%)")
    report.append(
        f"  Medium Risk (≥{CONFIG['thresholds']['medium_risk_score']}): {medium_risk:,} ({medium_risk/len(df)*100:.1f}%)")
    report.append(f"  Low Risk: {low_risk:,} ({low_risk/len(df)*100:.1f}%)")
    report.append("")

    # ID Anomalies
    id_anomalies = analyses.get('id_anomalies', {})
    report.append("VOTER ID ANOMALIES:")
    report.append(
        f"  Duplicate Voter IDs: {len(id_anomalies.get('duplicates', []))}")
    report.append(
        f"  Sequential ID Runs: {len(id_anomalies.get('sequential_runs', []))}")
    report.append(
        f"  Format Inconsistencies: {len(id_anomalies.get('format_inconsistencies', {}))}")
    report.append("")

    # Benford Analysis
    benford = analyses.get('benford_analysis', {})
    report.append("BENFORD'S LAW VIOLATIONS:")
    for field, results in benford.items():
        first_digit = results.get('first_digit_test', {})
        if first_digit.get('significant_deviation', False):
            report.append(
                f"  {field}: p-value = {first_digit.get('p_value', 0):.4f} (SUSPICIOUS)")
    report.append("")

    # Demographic Anomalies
    demo = analyses.get('demographic_anomalies', {})
    age_anomalies = demo.get('age_anomalies', {})
    if age_anomalies.get('heaping_suspicious', False):
        report.append("DEMOGRAPHIC ANOMALIES:")
        report.append(
            f"  Age Heaping: {age_anomalies.get('heaping_percentage', 0):.1f}% (SUSPICIOUS)")

    gender_anomalies = demo.get('gender_anomalies', {})
    if gender_anomalies.get('suspicious', False):
        report.append(
            f"  Gender Ratio Deviation: {gender_anomalies.get('deviation', 0):.3f} (SUSPICIOUS)")
    report.append("")

    # Top Risk Records
    top_risk = df_scored.nlargest(10, 'risk_score')[
        ['name', 'voter_id', 'risk_score', 'risk_flags']]
    report.append("TOP 10 HIGHEST RISK RECORDS:")
    for idx, row in top_risk.iterrows():
        flags_str = ', '.join(
            row['risk_flags']) if row['risk_flags'] else 'None'
        report.append(
            f"  {row['name'][:20]:<20} | {row['voter_id']:<15} | Score: {row['risk_score']:.3f} | Flags: {flags_str}")
    report.append("")

    report.append("="*80)

    return '\n'.join(report)


def call_groq_api(prompt: str, max_retries: int = 3) -> str:
    """Call Groq API to generate human-readable explanations using groq Python package"""

    if not CONFIG['groq_api_key']:
        print("Warning: GROQ_API_KEY not set. Skipping AI report generation.")
        return "AI report generation skipped - API key not configured."

    client = groq.Groq(
        api_key=GROQ_API_KEY)

    system_message = {
        "role": "system",
        "content": "You are an expert election analyst who explains complex data analysis findings in simple, clear language that ordinary citizens can understand. Focus on the implications and significance of the findings."
    }
    user_message = {
        "role": "user",
        "content": prompt
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CONFIG['groq_model'],
                messages=[system_message, user_message],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                import time
                time.sleep(2)

    return "Unable to generate AI explanation due to API issues."


def generate_groq_powered_report(df: pd.DataFrame, analyses: Dict, df_scored: pd.DataFrame) -> Dict:
    """Generate a comprehensive, AI-explained corruption report for public understanding"""

    print("Generating AI-powered public report using Groq...")

    # Prepare summary statistics for AI analysis
    total_records = len(df)
    high_risk = len(df_scored[df_scored['risk_score']
                    >= CONFIG['thresholds']['high_risk_score']])
    medium_risk = len(df_scored[(df_scored['risk_score'] >= CONFIG['thresholds']['medium_risk_score']) &
                                (df_scored['risk_score'] < CONFIG['thresholds']['high_risk_score'])])

    # Key findings summary
    key_findings = {
        'total_voters': total_records,
        'high_risk_count': high_risk,
        'high_risk_percentage': (high_risk / total_records) * 100,
        'medium_risk_count': medium_risk,
        'medium_risk_percentage': (medium_risk / total_records) * 100,
        'duplicate_voter_ids': len(analyses.get('id_anomalies', {}).get('duplicates', [])),
        'sequential_id_patterns': len(analyses.get('id_anomalies', {}).get('sequential_runs', [])),
        'suspicious_families': len(analyses.get('demographic_anomalies', {}).get('family_anomalies', {}).get('large_families', {})),
        'age_heaping_detected': analyses.get('demographic_anomalies', {}).get('age_anomalies', {}).get('heaping_suspicious', False),
        'age_heaping_percentage': analyses.get('demographic_anomalies', {}).get('age_anomalies', {}).get('heaping_percentage', 0),
        'benford_violations': sum(1 for field, results in analyses.get('benford_analysis', {}).items()
                                  if results.get('first_digit_test', {}).get('significant_deviation', False))
    }

    # Create prompts for different sections
    prompts = {
        'executive_summary': f"""
        Based on this election data analysis, explain in simple terms what we found:
        - Total voters analyzed: {key_findings['total_voters']:,}
        - High risk voters: {key_findings['high_risk_count']:,} ({key_findings['high_risk_percentage']:.1f}%)
        - Medium risk voters: {key_findings['medium_risk_count']:,} ({key_findings['medium_risk_percentage']:.1f}%)
        - Duplicate voter IDs found: {key_findings['duplicate_voter_ids']}
        - Suspicious voting patterns detected: {key_findings['sequential_id_patterns']}
        - Large suspicious families: {key_findings['suspicious_families']}
        - Age manipulation detected: {key_findings['age_heaping_detected']} ({key_findings['age_heaping_percentage']:.1f}% of ages end in 0 or 5)
        
        Explain what this means for election integrity and what citizens should know.
        """,

        'voter_id_issues': f"""
        We found these voter ID problems:
        - {key_findings['duplicate_voter_ids']} cases where multiple people have the same voter ID
        - {key_findings['sequential_id_patterns']} patterns where voter IDs appear to be artificially created in sequence
        
        Explain why these are serious problems and what they might indicate about the voter registration process.
        """,

        'demographic_concerns': f"""
        We discovered demographic anomalies:
        - Age heaping detected: {key_findings['age_heaping_percentage']:.1f}% of voters have ages ending in 0 or 5 (normal would be 20%)
        - {key_findings['suspicious_families']} families with unusually large numbers of voters at the same address
        - Statistical patterns that don't match natural population distributions
        
        Explain what these patterns suggest and why they're concerning for election integrity.
        """,

        'recommendations': f"""
        Based on these findings with {key_findings['high_risk_percentage']:.1f}% high-risk voters, what should be done?
        Provide practical recommendations for:
        1. Election officials
        2. Voters and citizens
        3. Policy makers
        4. Immediate actions needed
        
        Keep explanations simple and actionable.
        """
    }

    # Generate AI explanations for each section
    ai_explanations = {}
    for section, prompt in prompts.items():
        print(f"  Generating explanation for: {section}")
        ai_explanations[section] = call_groq_api(prompt)

    # Compile the complete public report
    public_report = {
        "report_metadata": {
            "title": "Election Data Integrity Analysis - Public Report",
            "generated_date": datetime.now().isoformat(),
            "analysis_method": "Statistical and Pattern Analysis using AI Explanation",
            "data_source": "Voter Registration Database",
            "total_records_analyzed": total_records
        },

        "executive_summary": {
            "overview": ai_explanations['executive_summary'],
            "key_statistics": {
                "total_voters": key_findings['total_voters'],
                "voters_flagged_high_risk": {
                    "count": key_findings['high_risk_count'],
                    "percentage": round(key_findings['high_risk_percentage'], 2)
                },
                "voters_flagged_medium_risk": {
                    "count": key_findings['medium_risk_count'],
                    "percentage": round(key_findings['medium_risk_percentage'], 2)
                },
                "overall_integrity_score": round(100 - key_findings['high_risk_percentage'] - (key_findings['medium_risk_percentage'] * 0.5), 1)
            }
        },

        "major_concerns_found": {
            "duplicate_voter_identities": {
                "description": "Multiple voters sharing the same ID number",
                "cases_found": key_findings['duplicate_voter_ids'],
                "explanation": ai_explanations['voter_id_issues'],
                "severity": "HIGH" if key_findings['duplicate_voter_ids'] > 10 else "MEDIUM" if key_findings['duplicate_voter_ids'] > 0 else "LOW"
            },

            "artificial_voter_creation": {
                "description": "Voter IDs created in suspicious sequential patterns",
                "patterns_found": key_findings['sequential_id_patterns'],
                "explanation": "Sequential voter ID patterns may indicate bulk creation of fake voter records",
                "severity": "HIGH" if key_findings['sequential_id_patterns'] > 5 else "MEDIUM" if key_findings['sequential_id_patterns'] > 0 else "LOW"
            },

            "demographic_manipulation": {
                "description": "Unnatural patterns in voter ages and family structures",
                "age_heaping_detected": key_findings['age_heaping_detected'],
                "age_heaping_percentage": round(key_findings['age_heaping_percentage'], 1),
                "suspicious_large_families": key_findings['suspicious_families'],
                "explanation": ai_explanations['demographic_concerns'],
                "severity": "HIGH" if key_findings['age_heaping_percentage'] > 50 else "MEDIUM" if key_findings['age_heaping_percentage'] > 30 else "LOW"
            },

            "statistical_anomalies": {
                "description": "Patterns that violate natural statistical distributions",
                "benford_law_violations": key_findings['benford_violations'],
                "explanation": "When numbers don't follow expected patterns, it often indicates artificial data creation",
                "severity": "MEDIUM" if key_findings['benford_violations'] > 2 else "LOW"
            }
        },

        "what_this_means_for_voters": {
            "impact_assessment": "Based on the analysis, here's what these findings mean for election integrity:",
            "voter_confidence_impact": ai_explanations['executive_summary'],
            "potential_consequences": [
                "Inflated voter rolls may affect constituency boundaries",
                "Fake voters could impact election outcomes",
                "Legitimate voters might be turned away if their IDs are duplicated",
                "Public trust in electoral process may be undermined"
            ]
        },

        "recommendations_and_next_steps": {
            "immediate_actions": ai_explanations['recommendations'],
            "for_election_officials": [
                "Audit all flagged high-risk voter records immediately",
                "Implement stronger verification processes",
                "Cross-reference voter data with other government databases",
                "Investigate source of duplicate and sequential IDs"
            ],
            "for_citizens": [
                "Verify your voter registration details are correct",
                "Report any suspicious activity you observe",
                "Stay informed about election integrity measures",
                "Participate in the democratic process despite concerns"
            ],
            "for_policymakers": [
                "Strengthen voter registration verification requirements",
                "Implement regular data integrity audits",
                "Enhance penalties for voter registration fraud",
                "Improve transparency in voter roll maintenance"
            ]
        },

        "technical_summary_simplified": {
            "methods_used": [
                "Statistical analysis to find unusual patterns",
                "Comparison with expected natural distributions",
                "Pattern recognition for artificial data creation",
                "Cross-referencing for duplicate entries"
            ],
            "confidence_level": "High - Multiple statistical methods confirm findings",
            "limitations": [
                "Analysis based on registration data only",
                "Some patterns may have legitimate explanations",
                "Requires human verification of flagged cases"
            ]
        },

        "appendix": {
            "high_risk_examples": df_scored[df_scored['risk_score'] >= CONFIG['thresholds']['high_risk_score']].head(5)[
                ['name', 'voter_id', 'age', 'house_number', 'risk_score']
            ].to_dict('records'),
            "glossary": {
                "age_heaping": "When ages cluster unnaturally around multiples of 5 (like 20, 25, 30), often indicating fake data",
                "benford_law": "A mathematical principle that numbers in natural datasets follow predictable patterns",
                "sequential_patterns": "When ID numbers appear in artificial consecutive order, suggesting bulk creation",
                "risk_score": "A calculated measure (0-1) of how likely a voter record is to be fraudulent"
            },
            "data_sources": "Voter registration database analysis",
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "contact_info": "For questions about this analysis, contact your local election commission"
        }
    }

    return public_report


def main():
    """Main analysis function"""
    print("Starting Enhanced Election Data Corruption Analysis...")

    # Load data
    df = load_voter_data(CONFIG['input_json'])
    if df.empty:
        print("Error: No data found or failed to load data")
        return

    print(f"Loaded {len(df):,} voter records")

    # Perform comprehensive analyses
    analyses = {}

    print("1. Analyzing Voter ID anomalies...")
    analyses['id_anomalies'] = detect_voter_id_anomalies(df)

    print("2. Performing Benford's Law analysis...")
    analyses['benford_analysis'] = advanced_benford_analysis(df)

    print("3. Detecting demographic anomalies...")
    analyses['demographic_anomalies'] = demographic_anomaly_detection(df)

    print("4. Analyzing name patterns...")
    analyses['name_patterns'] = name_pattern_analysis(df)

    print("5. Performing clustering analysis...")
    analyses['clustering'] = clustering_analysis(df)

    print("6. Calculating risk scores...")
    df_scored = calculate_risk_scores(df, analyses)

    # Save results
    output_data = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_records': len(df),
            'configuration': CONFIG
        },
        'analyses': analyses,
        # Limit output size
        'scored_records': df_scored.to_dict('records')[:1000]
    }

    with open(CONFIG['output_flags'], 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    # Save high-risk records to CSV
    high_risk_records = df_scored[df_scored['risk_score']
                                  >= CONFIG['thresholds']['medium_risk_score']]
    high_risk_records[['name', 'voter_id', 'age', 'house_number', 'risk_score', 'risk_flags']].to_csv(
        CONFIG['output_summary'], index=False, encoding='utf-8'
    )

    # Generate and save report
    report = generate_comprehensive_report(df, analyses, df_scored)
    with open(CONFIG['output_report'], 'w', encoding='utf-8') as f:
        f.write(report)

    # Generate Groq-powered public report
    print("7. Generating AI-powered public report...")
    public_report = generate_groq_powered_report(df, analyses, df_scored)

    with open(CONFIG['output_groq_report'], 'w', encoding='utf-8') as f:
        json.dump(public_report, f, ensure_ascii=False, indent=2, default=str)

    print(report)
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  - Detailed analysis: {CONFIG['output_flags']}")
    print(f"  - High-risk records: {CONFIG['output_summary']}")
    print(f"  - Technical report: {CONFIG['output_report']}")
    print(f"  - Public AI report: {CONFIG['output_groq_report']}")

    # Display public report summary
    print("\n" + "="*60)
    print("PUBLIC REPORT SUMMARY (AI-Generated)")
    print("="*60)
    print(
        f"Overall Integrity Score: {public_report['executive_summary']['key_statistics']['overall_integrity_score']}%")
    print(f"High Risk Voters: {public_report['executive_summary']['key_statistics']['voters_flagged_high_risk']['count']} ({public_report['executive_summary']['key_statistics']['voters_flagged_high_risk']['percentage']}%)")
    print(
        f"Major Concerns: {len([c for c in public_report['major_concerns_found'].values() if isinstance(c, dict) and c.get('severity') == 'HIGH'])}")
    print("="*60)


if __name__ == "__main__":
    main()
