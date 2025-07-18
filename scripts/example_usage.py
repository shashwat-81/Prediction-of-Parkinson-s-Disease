"""
Example script demonstrating how to use the Parkinson's disease prediction system
"""

import os
from predict_disease import ParkinsonsPredictor

def example_single_file_prediction():
    """Example of predicting a single audio file"""
    print("=" * 60)
    print("EXAMPLE 1: Single File Prediction")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = ParkinsonsPredictor()
    
    # Example with test data
    test_file = "../data/TestData/shashwat.wav"
    
    if os.path.exists(test_file):
        print(f"Analyzing test file: {test_file}")
        result = predictor.predict_single_file(test_file)
        predictor.print_detailed_report(result)
    else:
        print(f"Test file not found: {test_file}")
        print("Please update the path to point to your audio file")

def example_multiple_files_prediction():
    """Example of predicting multiple audio files"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple Files Prediction")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = ParkinsonsPredictor()
    
    # Example with test data folder
    test_folder = "../data/TestData"
    
    if os.path.exists(test_folder):
        print(f"Analyzing all files in folder: {test_folder}")
        results = predictor.predict_multiple_files(
            test_folder,
            output_csv="example_results.csv"
        )
        
        if results:
            print(f"\n📊 SUMMARY STATISTICS")
            print("-" * 30)
            
            # Calculate summary statistics
            total_files = len(results)
            high_risk_files = len([r for r in results if "High" in r.get('risk_level', '')])
            avg_parkinson_prob = sum([r.get('ensemble_parkinson_probability', r.get('parkinson_probability', 0)) 
                                    for r in results]) / total_files
            
            print(f"Total files analyzed: {total_files}")
            print(f"High/Very High risk files: {high_risk_files}")
            print(f"Average Parkinson's probability: {avg_parkinson_prob:.3f}")
            
            # Risk distribution
            risk_counts = {}
            for result in results:
                risk = result.get('risk_level', 'Unknown')
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print(f"\nRisk Distribution:")
            for risk, count in sorted(risk_counts.items()):
                percentage = (count / total_files) * 100
                print(f"  {risk}: {count} files ({percentage:.1f}%)")
                
    else:
        print(f"Test folder not found: {test_folder}")
        print("Please update the path to point to your audio folder")

def example_custom_audio_analysis():
    """Example showing how to analyze your own audio files"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Analyze Your Own Audio")
    print("=" * 60)
    
    print("To analyze your own audio files:")
    print("1. Record yourself saying 'Ah' for 3-5 seconds")
    print("2. Save as WAV file (e.g., 'my_voice.wav')")
    print("3. Use this code:")
    print()
    
    code_example = '''
# Initialize predictor
from predict_disease import ParkinsonsPredictor
predictor = ParkinsonsPredictor()

# Analyze your audio file
result = predictor.predict_single_file("path/to/your/audio.wav")
predictor.print_detailed_report(result)

# Or analyze multiple files
results = predictor.predict_multiple_files("path/to/audio/folder")
'''
    
    print(code_example)
    
    # Interactive example
    print("\n" + "─" * 40)
    user_file = input("Enter path to your audio file (or press Enter to skip): ").strip()
    
    if user_file and os.path.exists(user_file):
        try:
            predictor = ParkinsonsPredictor()
            result = predictor.predict_single_file(user_file)
            predictor.print_detailed_report(result)
        except Exception as e:
            print(f"Error analyzing your file: {e}")
    elif user_file:
        print(f"File not found: {user_file}")

def main():
    """Run all examples"""
    print("🎤 Parkinson's Disease Prediction Examples")
    print("=" * 60)
    
    try:
        # Example 1: Single file
        example_single_file_prediction()
        
        # Example 2: Multiple files
        example_multiple_files_prediction()
        
        # Example 3: Custom audio
        example_custom_audio_analysis()
        
        print("\n" + "=" * 60)
        print("✅ Examples completed successfully!")
        print("📋 Check 'example_results.csv' for detailed results")
        print("📖 See 'HOW_TO_USE.md' for detailed instructions")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure all model files are present and try again.")

if __name__ == "__main__":
    main()
