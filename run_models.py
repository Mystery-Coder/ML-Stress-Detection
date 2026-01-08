"""
Batch Audio Evaluation - Analyze entire folder and create presentation charts
"""
import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model # type: ignore
from pydub import AudioSegment
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter


def extract_mfcc(file_path, duration, sr, offset, n_mfcc):
    """Extract MFCC features from audio file"""
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast', 
                                     duration=duration, sr=sr, offset=offset)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc), axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None


def prepare_features(mfcc_features, required_length):
    """Pad or trim MFCC features to required length"""
    current_length = len(mfcc_features)
    if current_length < required_length:
        padded_array = np.pad(mfcc_features, (0, required_length - current_length), 
                            'constant', constant_values=0)
    else:
        padded_array = mfcc_features[:required_length]
    return np.expand_dims(padded_array, axis=0)


def split_audio_for_evaluation(audio_path, output_dir="temp_chunks"):
    """Split audio into 3-second and 2-minute chunks"""
    os.makedirs(output_dir, exist_ok=True)
    folder_3sec = os.path.join(output_dir, "3sec")
    folder_2min = os.path.join(output_dir, "2min")
    
    os.makedirs(folder_3sec, exist_ok=True)
    os.makedirs(folder_2min, exist_ok=True)
    
    audio = AudioSegment.from_file(audio_path)
    
    # Split into 3-second chunks
    chunk_duration_3sec = 3 * 1000
    for i in range(0, len(audio), chunk_duration_3sec):
        chunk = audio[i:i + chunk_duration_3sec]
        chunk_filename = f"chunk_{i // chunk_duration_3sec + 1}.wav"
        chunk.export(os.path.join(folder_3sec, chunk_filename), format="wav")
    
    # Split into 2-minute chunks
    chunk_duration_2min = 2 * 60 * 1000
    for i in range(0, len(audio), chunk_duration_2min):
        chunk = audio[i:i + chunk_duration_2min]
        chunk_filename = f"chunk_{i // chunk_duration_2min + 1}.wav"
        chunk.export(os.path.join(folder_2min, chunk_filename), format="wav")
    
    audio_duration_sec = len(audio) / 1000
    
    return folder_3sec, folder_2min, audio_duration_sec


def evaluate_single_file(audio_path, model_emotion, model_depression, lb_emo, lb_dp):
    """Evaluate a single audio file"""
    
    # Split audio
    folder_3sec, folder_2min, duration_sec = split_audio_for_evaluation(audio_path)
    
    # Predict emotions
    emotions = []
    files_3sec = sorted(os.listdir(folder_3sec))
    
    for file in files_3sec:
        try:
            file_path = os.path.join(folder_3sec, file)
            mfccs = extract_mfcc(file_path, duration=3, sr=44100, offset=0.5, n_mfcc=13)
            
            if mfccs is not None:
                x_testcnn = prepare_features(mfccs, 259)
                y_pred = model_emotion.predict(x_testcnn, verbose=0)
                predicted_class = np.argmax(y_pred, axis=1)
                predicted_emotion = lb_emo.inverse_transform(predicted_class)
                emotions.append(predicted_emotion[0])
        except Exception as e:
            print(f"  Error processing emotion chunk: {e}")
    
    # Predict depression
    depressions = []
    files_2min = sorted(os.listdir(folder_2min))
    
    for file in files_2min:
        try:
            file_path = os.path.join(folder_2min, file)
            mfccs = extract_mfcc(file_path, duration=2*60, sr=44100, offset=0.5, n_mfcc=20)
            
            if mfccs is not None:
                x_testcnn = prepare_features(mfccs, 10293)
                y_pred = model_depression.predict(x_testcnn, verbose=0)
                predicted_class = np.argmax(y_pred, axis=1)
                predicted_depression = lb_dp.inverse_transform(predicted_class)
                depressions.append(predicted_depression[0])
        except Exception as e:
            print(f"  Error processing depression chunk: {e}")
    
    return emotions, depressions, duration_sec


def create_aggregate_visualizations(all_emotions, all_depressions, total_minutes, 
                                   num_files, output_folder="results"):
    """Create presentation charts for the entire dataset"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING PRESENTATION CHARTS")
    print("="*60)
    
    # 1. Overall Emotion Distribution Pie Chart
    emotion_counts = Counter(all_emotions)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(emotion_counts)))
    plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(f'Emotion Distribution Across {total_minutes:.1f} Minutes of Audio Analysis', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/1_overall_emotion_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Overall Emotion Distribution (Pie Chart)")
    
    
    # 2. Emotion Frequency Bar Chart
    plt.figure(figsize=(12, 6))
    emotions_list = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    bars = plt.bar(emotions_list, counts, color='steelblue', edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title(f'Emotion Detection Frequency ({num_files} Audio Files Analyzed)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Emotion Type', fontsize=12)
    plt.ylabel('Number of Detections', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/2_emotion_frequency.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Emotion Frequency Bar Chart")
    
    
    # 3. Depression/Stress Level Distribution
    depression_counts = Counter(all_depressions)
    plt.figure(figsize=(10, 6))
    dep_levels = ['low', 'med', 'high']
    dep_counts = [depression_counts.get(level, 0) for level in dep_levels]
    colors_dep = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = plt.bar(dep_levels, dep_counts, color=colors_dep, edgecolor='black', 
                   width=0.6, linewidth=1.5)
    for bar, level in zip(bars, dep_levels):
        height = bar.get_height()
        if height > 0:
            percentage = (height / sum(dep_counts)) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title(f'Workplace Stress Level Analysis ({total_minutes:.1f} Minutes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Stress Level', fontsize=12)
    plt.ylabel('Number of Detections', fontsize=12)
    plt.ylim(0, max(dep_counts) * 1.25 if max(dep_counts) > 0 else 1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/3_stress_level_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Stress Level Distribution")
    
    
    # 4. Positive vs Negative Emotions Analysis
    positive_emotions = ['happy', 'calm', 'surprised']
    negative_emotions = ['sad', 'angry', 'fearful', 'disgust']
    
    positive_count = sum(emotion_counts.get(e, 0) for e in positive_emotions)
    negative_count = sum(emotion_counts.get(e, 0) for e in negative_emotions)
    neutral_count = emotion_counts.get('neutral', 0)
    
    plt.figure(figsize=(10, 7))
    categories = ['Positive\nEmotions', 'Neutral', 'Negative\nEmotions']
    values = [positive_count, neutral_count, negative_count]
    colors_cat = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    bars = plt.bar(categories, values, color=colors_cat, edgecolor='black', 
                   width=0.6, linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(all_emotions)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Workplace Emotional Climate Analysis', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Detections', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/4_emotional_climate.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Emotional Climate Analysis")
    
    
    # 5. Summary Statistics Box
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    summary_text = f"""
        WORKPLACE AUDIO ANALYSIS SUMMARY

        Total Audio Files Analyzed: {num_files}
        Total Duration Analyzed: {total_minutes:.1f} minutes ({total_minutes/60:.2f} hours)
        Total Emotion Predictions: {len(all_emotions)}
        Total Stress Assessments: {len(all_depressions)}

        EMOTION BREAKDOWN:
        """
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_emotions)) * 100
        summary_text += f"  • {emotion.capitalize():12s}: {count:4d} detections ({percentage:5.1f}%)\n"
    
    summary_text += f"\nSTRESS LEVEL BREAKDOWN:\n"
    for level in ['low', 'med', 'high']:
        count = depression_counts.get(level, 0)
        percentage = (count / len(all_depressions)) * 100 if len(all_depressions) > 0 else 0
        summary_text += f"  • {level.upper():12s}: {count:4d} detections ({percentage:5.1f}%)\n"
    
    summary_text += f"""
        WORKPLACE HEALTH INDICATORS:
        • Positive Emotions: {positive_count} ({positive_count/len(all_emotions)*100:.1f}%)
        • Negative Emotions: {negative_count} ({negative_count/len(all_emotions)*100:.1f}%)
        • Neutral State: {neutral_count} ({neutral_count/len(all_emotions)*100:.1f}%)
        

        """
    
    ax.text(0.5, 0.5, summary_text, fontsize=11, family='monospace',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(f"{output_folder}/5_analysis_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created: Analysis Summary")
    
    print("\n" + "="*60)


def evaluate_folder(audio_folder, model_dir="Model", lb_dir="lb", output_folder="results"):
    """
    Evaluate all audio files in a folder and create presentation charts
    
    Args:
        audio_folder: Folder containing audio files
        model_dir: Directory containing Keras models
        lb_dir: Directory containing label encoders
        output_folder: Where to save results and charts
    """
    
    print("="*60)
    print("BATCH AUDIO EVALUATION FOR PRESENTATION")
    print("="*60)
    
    # Get audio files
    audio_files = [f for f in os.listdir(audio_folder) 
                   if f.endswith(('.wav', '.mp3', '.webm', '.m4a'))]
    
    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return
    
    print(f"\nFound {len(audio_files)} audio files to analyze")
    
    # Load models
    print("\nLoading models...")
    try:
        lb_dp = pickle.load(open(os.path.join(lb_dir, "lb-depression.sav"), 'rb'))
        lb_emo = pickle.load(open(os.path.join(lb_dir, "lb-emotion.sav"), 'rb'))
        model_emotion = load_model(os.path.join(model_dir, "emotion.keras"))
        model_depression = load_model(os.path.join(model_dir, "depression.keras"))
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Process all files
    all_emotions = []
    all_depressions = []
    total_duration = 0
    file_results = {}
    
    print("\n" + "="*60)
    print("PROCESSING FILES")
    print("="*60)
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file}")
        audio_path = os.path.join(audio_folder, audio_file)
        
        try:
            emotions, depressions, duration_sec = evaluate_single_file(
                audio_path, model_emotion, model_depression, lb_emo, lb_dp
            )
            
            all_emotions.extend(emotions)
            all_depressions.extend(depressions)
            total_duration += duration_sec
            
            file_results[audio_file] = {
                'emotions': emotions,
                'depressions': depressions,
                'duration_sec': duration_sec
            }
            
            print(f"  ✓ Completed: {len(emotions)} emotion predictions, "
                  f"{len(depressions)} stress predictions ({duration_sec/60:.1f} min)")
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_file}: {e}")
            continue
    
    total_minutes = total_duration / 60
    
    # Create visualizations
    create_aggregate_visualizations(all_emotions, all_depressions, total_minutes, 
                                   len(audio_files), output_folder)
    
    # Save detailed results
    results = {
        "total_files": len(audio_files),
        "total_duration_minutes": total_minutes,
        "total_emotions_detected": len(all_emotions),
        "total_stress_assessments": len(all_depressions),
        "emotion_summary": dict(Counter(all_emotions)),
        "depression_summary": dict(Counter(all_depressions)),
        "per_file_results": file_results
    }
    
    with open(f"{output_folder}/detailed_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTotal Files Analyzed: {len(audio_files)}")
    print(f"Total Duration: {total_minutes:.1f} minutes ({total_minutes/60:.2f} hours)")
    print(f"Total Predictions: {len(all_emotions)} emotions, {len(all_depressions)} stress levels")
    print(f"\nResults saved to: {output_folder}/")
    print(f"  - 5 presentation charts (PNG)")
    print(f"  - Detailed results (JSON)")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_folder.py <audio_folder> [output_folder]")
        print("\nExample:")
        print("  python evaluate_folder.py 6min_test_files")
        print("  python evaluate_folder.py my_audio_files presentation_results")
        sys.exit(1)
    
    audio_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "results"
    
    evaluate_folder(audio_folder, output_folder=output_folder)
