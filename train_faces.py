"""
SENTINEL v5.0 - Face Database Training
Create InsightFace embeddings from face images
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from PIL import Image
import glob


def train_face_database(dataset_dir: str = "dataset", output_file: str = "face_database.pkl"):
    """
    Train face database from images
    
    Args:
        dataset_dir: Directory containing subdirectories with person images
                    Structure: dataset/person_name/*.jpg
        output_file: Output pickle file path
    """
    print("=" * 80)
    print("SENTINEL v5.0 - Face Database Training")
    print("=" * 80)
    
    # Initialize InsightFace
    print("\n[1/4] Initializing InsightFace (buffalo_l model)...")
    try:
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0)
        print("✓ InsightFace initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize InsightFace: {e}")
        return False
    
    # Scan dataset directory
    print(f"\n[2/4] Scanning dataset directory: {dataset_dir}")
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"✗ Dataset directory not found: {dataset_path}")
        return False
    
    # Find all person directories
    person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(person_dirs) == 0:
        print("✗ No person directories found in dataset")
        print(f"   Expected structure: {dataset_dir}/person_name/*.jpg")
        return False
    
    print(f"✓ Found {len(person_dirs)} person(s) in dataset:")
    for person_dir in person_dirs:
        image_count = len(list(person_dir.glob('*')))
        print(f"   - {person_dir.name}: {image_count} images")
    
    # Process each person
    print(f"\n[3/4] Processing face images...")
    database = {}
    
    for person_dir in person_dirs:
        person_name = person_dir.name
        print(f"\nProcessing: {person_name}")
        print("-" * 40)
        
        embeddings = []
        image_files = list(person_dir.glob('*'))
        
        for i, img_file in enumerate(image_files):
            try:
                # Skip if not an image
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                # Load image
                img = np.array(Image.open(img_file).convert('RGB'))
                
                # Detect faces
                faces = app.get(img)
                
                if len(faces) == 0:
                    print(f"   [{i+1}/{len(image_files)}] ✗ {img_file.name} - No face detected")
                    continue
                
                # Get largest face (in case multiple)
                largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                embedding = largest_face.embedding
                
                embeddings.append(embedding)
                print(f"   [{i+1}/{len(image_files)}] ✓ {img_file.name}")
                
            except Exception as e:
                print(f"   [{i+1}/{len(image_files)}] ✗ {img_file.name} - Error: {e}")
        
        if len(embeddings) == 0:
            print(f"   WARNING: No valid faces extracted for {person_name}")
        else:
            database[person_name] = embeddings
            print(f"   SUCCESS: Extracted {len(embeddings)} face embeddings for {person_name}")
    
    # Save database
    print(f"\n[4/4] Saving face database...")
    
    if len(database) == 0:
        print("✗ No faces to save. Database is empty.")
        return False
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(database, f)
        
        print(f"✓ Database saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Failed to save database: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total persons: {len(database)}")
    for name, embeddings in database.items():
        print(f"  - {name}: {len(embeddings)} embeddings")
    print("\nYou can now run the SENTINEL system!")
    print("=" * 80)
    
    return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train SENTINEL face database from images"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="face_database.pkl",
        help="Output database file (default: face_database.pkl)"
    )
    
    args = parser.parse_args()
    
    success = train_face_database(args.dataset, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
