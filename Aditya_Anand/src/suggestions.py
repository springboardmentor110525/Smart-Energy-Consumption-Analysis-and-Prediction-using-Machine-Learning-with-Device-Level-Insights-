"""
Module 7: Smart Suggestions Engine
Week 7-8 Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SmartSuggestionsEngine:
    """
    Generate smart energy-saving suggestions based on consumption patterns
    """
    
    def __init__(self, df):
        """
        Initialize suggestions engine
        
        Args:
            df: DataFrame with energy consumption data
        """
        self.df = df
        self.suggestions = []
        self.device_stats = {}
        
    def analyze_device_usage(self, device_columns):
        """
        Analyze usage patterns for each device
        
        Args:
            device_columns: List of device column names
        """
        print("\n[INFO] Analyzing device usage patterns...")
        
        for device in device_columns:
            if device in self.df.columns:
                stats = {
                    'total_consumption': self.df[device].sum(),
                    'avg_consumption': self.df[device].mean(),
                    'max_consumption': self.df[device].max(),
                    'std_consumption': self.df[device].std(),
                    'active_hours': (self.df[device] > 0).sum(),
                    'peak_hour': self.df.groupby(self.df.index.hour)[device].mean().idxmax()
                }
                self.device_stats[device] = stats
        
        print(f"   [OK] Analyzed {len(self.device_stats)} devices")
        
        return self.device_stats
    
    def identify_high_consumers(self, threshold_percentile=75):
        """
        Identify devices with high energy consumption
        
        Args:
            threshold_percentile: Percentile threshold for high consumption
        """
        print(f"\n[INFO] Identifying high energy consumers (>{threshold_percentile}th percentile)...")
        
        total_consumptions = {device: stats['total_consumption'] 
                            for device, stats in self.device_stats.items()}
        
        threshold = np.percentile(list(total_consumptions.values()), threshold_percentile)
        
        high_consumers = {device: consumption 
                         for device, consumption in total_consumptions.items()
                         if consumption > threshold}
        
        print(f"   [OK] Found {len(high_consumers)} high-consuming devices")
        
        return high_consumers
    
    def detect_peak_usage_times(self):
        """Detect peak usage times for optimization"""
        print("\n[INFO] Detecting peak usage times...")
        
        # Hourly aggregation
        hourly_usage = self.df.groupby(self.df.index.hour).sum()
        
        # Find peak hours
        numeric_cols = hourly_usage.select_dtypes(include=[np.number]).columns
        total_hourly = hourly_usage[numeric_cols].sum(axis=1)
        
        peak_hours = total_hourly.nlargest(5).index.tolist()
        off_peak_hours = total_hourly.nsmallest(5).index.tolist()
        
        print(f"   [OK] Peak hours: {peak_hours}")
        print(f"   [OK] Off-peak hours: {off_peak_hours}")
        
        return peak_hours, off_peak_hours
    
    def generate_device_suggestions(self, device, stats, energy_tips):
        """
        Generate suggestions for a specific device
        
        Args:
            device: Device name
            stats: Device statistics
            energy_tips: Dictionary of energy-saving tips
        """
        suggestions = []
        
        # High consumption suggestion
        if stats['total_consumption'] > stats['avg_consumption'] * 1.5:
            tip = energy_tips.get(device, f"Consider optimizing {device} usage.")
            suggestions.append({
                'device': device,
                'type': 'High Consumption',
                'severity': 'High',
                'suggestion': tip,
                'potential_savings': f"{stats['total_consumption'] * 0.2:.2f} kWh"
            })
        
        # Peak hour usage
        if stats['peak_hour'] in [18, 19, 20, 21]:  # Evening peak hours
            suggestions.append({
                'device': device,
                'type': 'Peak Hour Usage',
                'severity': 'Medium',
                'suggestion': f"Consider shifting {device} usage from peak hour {stats['peak_hour']}:00 to off-peak hours.",
                'potential_savings': f"{stats['avg_consumption'] * 0.15:.2f} kWh"
            })
        
        # Always-on detection
        if stats['active_hours'] > len(self.df) * 0.9:
            suggestions.append({
                'device': device,
                'type': 'Always-On Device',
                'severity': 'Medium',
                'suggestion': f"{device} appears to be always on. Consider using timers or smart plugs.",
                'potential_savings': f"{stats['total_consumption'] * 0.1:.2f} kWh"
            })
        
        return suggestions
    
    def generate_all_suggestions(self, energy_tips):
        """
        Generate all energy-saving suggestions
        
        Args:
            energy_tips: Dictionary of energy-saving tips per device
        """
        print("\n[INFO] Generating smart energy-saving suggestions...")
        
        all_suggestions = []
        
        for device, stats in self.device_stats.items():
            device_suggestions = self.generate_device_suggestions(device, stats, energy_tips)
            all_suggestions.extend(device_suggestions)
        
        # Sort by severity
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        all_suggestions.sort(key=lambda x: severity_order[x['severity']])
        
        self.suggestions = all_suggestions
        
        print(f"   [OK] Generated {len(all_suggestions)} suggestions")
        
        return all_suggestions
    
    def calculate_potential_savings(self):
        """Calculate total potential energy savings"""
        total_savings = 0
        
        for suggestion in self.suggestions:
            savings_str = suggestion['potential_savings'].replace(' kWh', '')
            total_savings += float(savings_str)
        
        return total_savings
    
    def generate_report(self, save_path='reports/results/energy_suggestions.csv'):
        """Generate and save suggestions report"""
        print(f"\n[INFO] Generating suggestions report...")
        
        if not self.suggestions:
            print("   [WARN] No suggestions to report!")
            return
        
        # Create DataFrame
        df_suggestions = pd.DataFrame(self.suggestions)
        
        # Save to CSV
        df_suggestions.to_csv(save_path, index=False)
        
        print(f"   [OK] Report saved to {save_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("   ENERGY SAVINGS SUMMARY")
        print("="*60)
        print(f"   Total Suggestions: {len(self.suggestions)}")
        print(f"   High Priority: {sum(1 for s in self.suggestions if s['severity'] == 'High')}")
        print(f"   Medium Priority: {sum(1 for s in self.suggestions if s['severity'] == 'Medium')}")
        print(f"   Potential Total Savings: {self.calculate_potential_savings():.2f} kWh")
        print("="*60)
        
        # Print top 5 suggestions
        print("\n   TOP 5 SUGGESTIONS:")
        for i, suggestion in enumerate(self.suggestions[:5], 1):
            print(f"\n   {i}. [{suggestion['severity']}] {suggestion['device']}")
            print(f"      {suggestion['suggestion']}")
            print(f"      Potential Savings: {suggestion['potential_savings']}")
        
        return df_suggestions
    
    def get_daily_tips(self):
        """Get daily energy-saving tips"""
        daily_tips = [
            "[TIP] Turn off lights when leaving a room to save energy.",
            "[TIP] Set your thermostat 2-3 degrees lower in winter and higher in summer.",
            "[TIP] Unplug devices when not in use to avoid phantom power drain.",
            "[TIP] Take shorter showers to reduce water heating costs.",
            "[TIP] Run dishwasher and washing machine only with full loads.",
            "[TIP] Keep refrigerator temperature at 37-40°F for optimal efficiency.",
            "[TIP] Enable power-saving mode on computers and monitors.",
            "[TIP] Use natural light during the day instead of artificial lighting.",
            "[TIP] Charge devices during off-peak hours for lower rates.",
            "[TIP] Seal air leaks around windows and doors to improve insulation."
        ]
        
        # Return a random tip
        return np.random.choice(daily_tips)


def main():
    """Main execution function"""
    print("="*60)
    print("SMART ENERGY ANALYSIS - SUGGESTIONS ENGINE")
    print("="*60)
    
    # Load data
    print("\n[INFO] Loading data...")
    df = pd.read_csv('data/processed/energy_data_clean.csv',
                     index_col='time', parse_dates=True)
    print(f"   [OK] Loaded {len(df):,} records")
    
    # Initialize suggestions engine
    engine = SmartSuggestionsEngine(df)
    
    # Define device columns
    device_columns = [col for col in df.columns if any(device in col.lower()
                     for device in ['dishwasher', 'fridge', 'microwave', 'furnace',
                                   'kitchen', 'office', 'living', 'barn', 'well',
                                   'charger', 'heater', 'conditioning', 'theater',
                                   'lights', 'laundry', 'pump'])]
    
    print(f"\n   Found {len(device_columns)} devices to analyze")
    
    # Analyze device usage
    device_stats = engine.analyze_device_usage(device_columns)
    
    # Identify high consumers
    high_consumers = engine.identify_high_consumers(threshold_percentile=75)
    
    # Detect peak usage times
    peak_hours, off_peak_hours = engine.detect_peak_usage_times()
    
    # Energy tips dictionary
    energy_tips = {
        'Dishwasher': 'Run dishwasher only when full. Use eco mode if available.',
        'Fridge': 'Keep refrigerator at optimal temperature (37-40°F). Clean coils regularly.',
        'Microwave': 'Use microwave instead of oven for small meals to save energy.',
        'Air conditioning [kW]': 'Set AC to 24-26°C. Use programmable thermostat.',
        'Water heater [kW]': 'Lower water heater temperature to 120°F. Insulate tank.',
        'Furnace': 'Regular maintenance and filter changes improve efficiency.',
        'Pool Pump [kW]': 'Run pool pump during off-peak hours. Consider variable speed pump.',
        'Laundry [kW]': 'Wash clothes in cold water. Run full loads only.',
        'Car charger [kW]': 'Charge during off-peak hours for lower rates.',
        'Outdoor lights [kW]': 'Use LED bulbs and motion sensors. Install timers.',
        'Home Theater [kW]': 'Enable power-saving mode. Unplug when not in use.',
        'Living room': 'Use LED bulbs. Turn off lights when leaving room.',
        'Home office': 'Enable sleep mode on computers. Unplug chargers when not in use.',
        'Kitchen': 'Use energy-efficient appliances. Avoid opening oven door frequently.'
    }
    
    # Generate suggestions
    suggestions = engine.generate_all_suggestions(energy_tips)
    
    # Generate report
    df_suggestions = engine.generate_report()
    
    # Get daily tip
    daily_tip = engine.get_daily_tips()
    print(f"\n   [TIP] Daily Tip: {daily_tip}")
    
    print("\n" + "="*60)
    print("[OK] SUGGESTIONS ENGINE COMPLETE!")
    print("="*60)
    
    return engine, suggestions


if __name__ == "__main__":
    engine, suggestions = main()
