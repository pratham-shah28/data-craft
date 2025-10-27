import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from utils import setup_logging, load_config, ensure_dir
from schema_detector import SchemaDetector

class BiasDetector:
    def __init__(self, dataset_name):
        self.logger = setup_logging("bias_detection")
        self.config = load_config()
        self.dataset_name = dataset_name
        self.bias_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_name": dataset_name,
            "analysis": {},
            "summary": {}
        }
        
        # Load schema profile to get protected attributes
        detector = SchemaDetector()
        self.schema_profile = detector.load_schema_profile(dataset_name)
    
    def identify_protected_attributes(self, df):
        """Identify protected attributes from schema profile or auto-detect"""
        self.logger.info("Identifying protected attributes...")
        
        if self.schema_profile and 'protected_attributes' in self.schema_profile:
            protected_attrs = self.schema_profile['protected_attributes']
            self.logger.info(f"Using protected attributes from schema profile: {protected_attrs}")
        else:
            # Fallback: Auto-detect categorical columns
            self.logger.warning("No schema profile found. Auto-detecting categorical columns...")
            protected_attrs = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    unique_count = df[col].nunique()
                    if 2 <= unique_count <= 20:
                        protected_attrs.append(col)
        
        # Filter to only existing columns in dataframe
        protected_attrs = [attr for attr in protected_attrs if attr in df.columns]
        
        self.logger.info(f"Protected attributes to analyze: {protected_attrs}")
        return protected_attrs
    
    def identify_metrics(self, df):
        """Identify numerical metrics for bias analysis"""
        self.logger.info("Identifying numerical metrics...")
        
        if self.schema_profile and 'recommended_metrics' in self.schema_profile:
            metrics = self.schema_profile['recommended_metrics']
            self.logger.info(f"Using metrics from schema profile: {metrics}")
        else:
            # Fallback: Use all numerical columns
            self.logger.warning("No schema profile found. Using all numerical columns...")
            metrics = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Filter out ID columns (high uniqueness)
            metrics = [col for col in metrics if df[col].nunique() / len(df) < 0.95]
        
        # Filter to only existing columns
        metrics = [metric for metric in metrics if metric in df.columns]
        
        self.logger.info(f"Metrics to analyze: {metrics}")
        return metrics
    
    def analyze_group_statistics(self, df, attribute, metric):
        """Analyze statistics across different groups"""
        self.logger.info(f"Analyzing {metric} across {attribute}...")
        
        try:
            group_stats = df.groupby(attribute)[metric].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            
            return group_stats.to_dict('index')
        except Exception as e:
            self.logger.error(f"Error analyzing group statistics: {str(e)}")
            return None
    
    def statistical_parity_test(self, df, attribute, metric):
        """Test for statistical parity across groups using ANOVA"""
        try:
            groups = df.groupby(attribute)[metric].apply(list)
            
            # Perform ANOVA test
            group_values = [group for group in groups if len(group) > 1]
            
            if len(group_values) < 2:
                self.logger.warning(f"Not enough groups for statistical test: {attribute}")
                return None
            
            f_stat, p_value = stats.f_oneway(*group_values)
            
            return {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant_difference": bool(p_value < 0.05),
                "interpretation": "Significant difference detected between groups" if p_value < 0.05 else "No significant difference between groups"
            }
        except Exception as e:
            self.logger.error(f"Error in statistical parity test: {str(e)}")
            return None
    
    def disparate_impact_ratio(self, df, attribute, metric):
        """Calculate disparate impact ratio (80% rule)"""
        try:
            group_means = df.groupby(attribute)[metric].mean()
            
            if len(group_means) < 2:
                self.logger.warning(f"Not enough groups for disparate impact: {attribute}")
                return None
            
            max_mean = group_means.max()
            min_mean = group_means.min()
            
            # Disparate impact ratio (should be > 0.8 for fairness)
            ratio = min_mean / max_mean if max_mean > 0 else 0
            
            return {
                "ratio": float(ratio),
                "passes_80_percent_rule": bool(ratio >= 0.8),
                "privileged_group": str(group_means.idxmax()),
                "underprivileged_group": str(group_means.idxmin()),
                "privileged_mean": float(max_mean),
                "underprivileged_mean": float(min_mean),
                "interpretation": f"Ratio of {ratio:.2f} " + ("passes" if ratio >= 0.8 else "fails") + " the 80% rule"
            }
        except Exception as e:
            self.logger.error(f"Error in disparate impact calculation: {str(e)}")
            return None
    
    def calculate_coefficient_of_variation(self, df, attribute, metric):
        """Calculate coefficient of variation across groups"""
        try:
            group_means = df.groupby(attribute)[metric].mean()
            group_std = df.groupby(attribute)[metric].std()
            
            # Coefficient of variation for each group
            cv = (group_std / group_means).dropna()
            
            return {
                "mean_cv": float(cv.mean()),
                "max_cv": float(cv.max()),
                "min_cv": float(cv.min()),
                "cv_by_group": {str(k): float(v) for k, v in cv.items()},
                "interpretation": "Low CV indicates consistent performance across groups"
            }
        except Exception as e:
            self.logger.error(f"Error calculating coefficient of variation: {str(e)}")
            return None
    
    def detect_representation_bias(self, df, attribute):
        """Detect if certain groups are under/over-represented"""
        try:
            value_counts = df[attribute].value_counts()
            total = len(df)
            
            representation = {}
            for group, count in value_counts.items():
                percentage = (count / total) * 100
                representation[str(group)] = {
                    "count": int(count),
                    "percentage": float(percentage),
                    "under_represented": bool(percentage < 10)  # Less than 10% is concerning
                }
            
            under_represented_groups = [k for k, v in representation.items() if v['under_represented']]
            
            return {
                "representation_by_group": representation,
                "under_represented_groups": under_represented_groups,
                "has_representation_bias": bool(len(under_represented_groups) > 0),
                "interpretation": f"{len(under_represented_groups)} group(s) under-represented (< 10%)"
            }
        except Exception as e:
            self.logger.error(f"Error detecting representation bias: {str(e)}")
            return None
    
    def detect_bias(self):
        """Run comprehensive bias detection analysis"""
        self.logger.info(f"Starting bias detection for: {self.dataset_name}")
        
        try:
            # Load validated data
            data_path = Path(self.config['data']['processed_path']) / f"{self.dataset_name}_validated.csv"
            df = pd.read_csv(data_path)
            
            # Identify protected attributes and metrics
            protected_attributes = self.identify_protected_attributes(df)
            metrics = self.identify_metrics(df)
            
            if not protected_attributes:
                self.logger.warning("No protected attributes found. Skipping bias analysis.")
                self.bias_report['summary']['status'] = "skipped"
                self.bias_report['summary']['reason'] = "No protected attributes detected"
                return self.bias_report
            
            if not metrics:
                self.logger.warning("No numerical metrics found. Skipping bias analysis.")
                self.bias_report['summary']['status'] = "skipped"
                self.bias_report['summary']['reason'] = "No numerical metrics detected"
                return self.bias_report
            
            self.bias_report['protected_attributes'] = protected_attributes
            self.bias_report['metrics_analyzed'] = metrics
            
            bias_flags = 0
            total_tests = 0
            
            # Analyze each protected attribute
            for attribute in protected_attributes:
                if attribute not in df.columns:
                    self.logger.warning(f"Protected attribute '{attribute}' not found in data")
                    continue
                
                self.bias_report['analysis'][attribute] = {
                    "representation_bias": self.detect_representation_bias(df, attribute),
                    "metrics": {}
                }
                
                # Check representation bias
                if self.bias_report['analysis'][attribute]['representation_bias']:
                    if self.bias_report['analysis'][attribute]['representation_bias']['has_representation_bias']:
                        bias_flags += 1
                        self.logger.warning(f"⚠ Representation bias detected in {attribute}")
                
                # Analyze each metric for this attribute
                for metric in metrics:
                    if metric not in df.columns:
                        self.logger.warning(f"Metric '{metric}' not found in data")
                        continue
                    
                    self.logger.info(f"Analyzing bias: {attribute} vs {metric}")
                    total_tests += 1
                    
                    # Group statistics
                    group_stats = self.analyze_group_statistics(df, attribute, metric)
                    
                    # Statistical parity test
                    parity_test = self.statistical_parity_test(df, attribute, metric)
                    
                    # Disparate impact ratio
                    impact_ratio = self.disparate_impact_ratio(df, attribute, metric)
                    
                    # Coefficient of variation
                    cv_analysis = self.calculate_coefficient_of_variation(df, attribute, metric)
                    
                    self.bias_report['analysis'][attribute]['metrics'][metric] = {
                        "group_statistics": group_stats,
                        "statistical_parity_test": parity_test,
                        "disparate_impact": impact_ratio,
                        "coefficient_of_variation": cv_analysis
                    }
                    
                    # Check for bias flags
                    if impact_ratio and not impact_ratio['passes_80_percent_rule']:
                        bias_flags += 1
                        self.logger.warning(
                            f"⚠ Potential bias detected: {attribute} vs {metric} "
                            f"(Disparate Impact Ratio: {impact_ratio['ratio']:.3f})"
                        )
                    
                    if parity_test and parity_test['significant_difference']:
                        self.logger.info(
                            f"ℹ Significant difference detected: {attribute} vs {metric} "
                            f"(p-value: {parity_test['p_value']:.4f})"
                        )
            
            # Generate summary
            self.bias_report['summary'] = {
                "status": "completed",
                "total_tests": total_tests,
                "bias_flags": bias_flags,
                "bias_flag_percentage": float((bias_flags / total_tests * 100) if total_tests > 0 else 0),
                "overall_assessment": self._generate_assessment(bias_flags, total_tests),
                "recommendations": self._generate_recommendations(bias_flags, total_tests)
            }
            
            # Save bias report
            report_path = Path(self.config['data']['processed_path']) / f"{self.dataset_name}_bias_report.json"
            
            with open(report_path, 'w') as f:
                json.dump(self.bias_report, f, indent=2)
            
            self.logger.info(f"Bias report saved to {report_path}")
            self.logger.info(f"✓ Bias detection completed: {bias_flags} bias flags out of {total_tests} tests")
            
            return self.bias_report
            
        except Exception as e:
            self.logger.error(f"Error in bias detection: {str(e)}")
            raise
    
    def _generate_assessment(self, bias_flags, total_tests):
        """Generate overall bias assessment"""
        if total_tests == 0:
            return "No tests performed"
        
        bias_percentage = (bias_flags / total_tests) * 100
        
        if bias_percentage == 0:
            return "✓ No significant bias detected"
        elif bias_percentage < 20:
            return "⚠ Minor bias concerns detected"
        elif bias_percentage < 50:
            return "⚠⚠ Moderate bias detected - review recommended"
        else:
            return "⚠⚠⚠ Significant bias detected - immediate action required"
    
    def _generate_recommendations(self, bias_flags, total_tests):
        """Generate actionable recommendations"""
        recommendations = []
        
        if bias_flags == 0:
            recommendations.append("Continue monitoring for bias in future data updates")
        else:
            recommendations.append("Review groups with disparate impact ratios below 0.8")
            recommendations.append("Consider data augmentation for under-represented groups")
            recommendations.append("Investigate root causes of statistical differences")
            recommendations.append("Implement fairness constraints in downstream models")
            recommendations.append("Document bias mitigation strategies")
        
        return recommendations

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bias_detection.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    detector = BiasDetector(dataset_name)
    detector.detect_bias()