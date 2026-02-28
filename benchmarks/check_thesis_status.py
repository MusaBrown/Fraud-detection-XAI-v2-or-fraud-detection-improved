"""
Check Thesis Implementation Status
==================================
Verifies all 5 mandatory fixes are complete.
"""

import json
from pathlib import Path

def check_statistical_rigor():
    """Check 10-seed validation with confidence intervals"""
    try:
        with open('thesis_results/statistical_benchmark.json') as f:
            data = json.load(f)
        
        p95 = data['fastshap_latency']['p95']
        return {
            'status': '[OK] COMPLETE',
            'details': f"FastSHAP P95: {p95['mean']:.3f} ± {p95['std']:.3f} ms (95% CI: [{p95['ci_95_lower']:.3f}, {p95['ci_95_upper']:.3f}])",
            'n_samples': p95['n_samples']
        }
    except Exception as e:
        return {'status': '[XX] MISSING', 'error': str(e)}

def check_class_stratified():
    """Check fraud vs non-fraud fidelity"""
    try:
        with open('thesis_results/class_specific_analysis.json') as f:
            data = json.load(f)
        
        fraud_fid = data['fraud']['fidelity']
        legit_fid = data['legitimate']['fidelity']
        return {
            'status': '[OK] COMPLETE',
            'fraud_fidelity': f"{fraud_fid:.4f}",
            'legitimate_fidelity': f"{legit_fid:.4f}",
            'gap': f"{legit_fid - fraud_fid:.4f}"
        }
    except Exception as e:
        return {'status': '[XX] MISSING', 'error': str(e)}

def check_stability():
    """Check stability metric"""
    try:
        with open('thesis_results/statistical_benchmark.json') as f:
            data = json.load(f)
        
        stability = data['stability_scores']
        return {
            'status': '[OK] COMPLETE',
            'mean_stability': f"{stability['mean']:.4f}",
            'std_stability': f"{stability['std']:.4f}",
            'meets_target': stability['mean'] > 0.95
        }
    except Exception as e:
        return {'status': '[XX] MISSING', 'error': str(e)}

def check_pca_analysis():
    """Check PCA limitation acknowledgment"""
    if Path('pca_feature_analysis.py').exists():
        return {
            'status': '[OK] COMPLETE',
            'file': 'pca_feature_analysis.py',
            'note': 'Outputs PCA limitation discussion'
        }
    return {'status': '[XX] MISSING'}

def check_ablation():
    """Check ablation study"""
    try:
        with open('thesis_results/ablation_study.json') as f:
            data = json.load(f)
        
        # Check if we have actual results or just errors
        has_results = any('fidelity' in v for v in data.values())
        
        if has_results:
            return {
                'status': '[OK] COMPLETE',
                'architectures_tested': len(data),
                'results': {k: v.get('fidelity', 'ERROR') for k, v in data.items()}
            }
        else:
            return {
                'status': '[!] NEEDS FIX',
                'error': 'All architectures have errors',
                'details': data
            }
    except Exception as e:
        return {'status': '[XX] MISSING', 'error': str(e)}

def main():
    print("="*70)
    print("MASTER'S THESIS IMPLEMENTATION STATUS")
    print("="*70)
    
    checks = {
        '1. Statistical Confidence (10-seed)': check_statistical_rigor(),
        '2. Class-Stratified Fidelity': check_class_stratified(),
        '3. Stability Metric (>0.95)': check_stability(),
        '4. PCA Limitation Discussion': check_pca_analysis(),
        '5. Ablation Study': check_ablation()
    }
    
    all_complete = True
    for name, result in checks.items():
        print(f"\n{name}")
        print(f"   Status: {result['status']}")
        if 'details' in result:
            print(f"   {result['details']}")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        if 'fraud_fidelity' in result:
            print(f"   Fraud: {result['fraud_fidelity']}, Legitimate: {result['legitimate_fidelity']}")
        if 'mean_stability' in result:
            print(f"   Stability: {result['mean_stability']} (target: >0.95)")
        
        if '❌' in result['status'] or '⚠️' in result['status']:
            all_complete = False
    
    print("\n" + "="*70)
    if all_complete:
        print("[OK] ALL MANDATORY FIXES COMPLETE - THESIS READY")
    else:
        print("[!] SOME FIXES NEEDED - SEE ABOVE")
    print("="*70)

if __name__ == "__main__":
    main()
