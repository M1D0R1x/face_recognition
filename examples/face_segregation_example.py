#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script showing how to use the face segregation feature programmatically.

This demonstrates the API-level usage of face segregation without using the CLI.
"""

import face_recognition
import os
from collections import defaultdict
import numpy as np

def segregate_faces_example(input_folder, output_folder, tolerance=0.6):
    """
    Example implementation of face segregation using the face_recognition API.
    
    This shows how you can use the face_recognition library to:
    1. Load images from a folder
    2. Detect faces in each image
    3. Compare face encodings to group similar faces
    4. Organize faces into separate folders
    
    Args:
        input_folder: Path to folder containing images to process
        output_folder: Path to output folder for segregated faces
        tolerance: Distance threshold for face matching (default 0.6)
    """
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Storage for known faces
    known_face_encodings = []
    person_ids = []
    person_counter = 0
    
    # Get all image files (simple version)
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(image_files)} images...")
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        print(f"Processing {image_file}...")
        
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Find all faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            print(f"  No faces found in {image_file}")
            continue
            
        print(f"  Found {len(face_encodings)} face(s)")
        
        # Process each face
        for idx, face_encoding in enumerate(face_encodings):
            # Compare with known faces
            if not known_face_encodings:
                # First face - create new person
                person_id = person_counter
                person_counter += 1
                known_face_encodings.append(face_encoding)
                person_ids.append(person_id)
            else:
                # Compare with existing faces
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_idx = np.argmin(distances)
                
                if distances[best_match_idx] <= tolerance:
                    # Match found - use existing person ID
                    person_id = person_ids[best_match_idx]
                else:
                    # No match - create new person
                    person_id = person_counter
                    person_counter += 1
                    known_face_encodings.append(face_encoding)
                    person_ids.append(person_id)
            
            # Create person folder if needed
            person_folder = os.path.join(output_folder, f"person_{person_id}")
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)
            
            print(f"  Face {idx} -> person_{person_id}")
    
    print(f"\nSegregation complete! Found {person_counter} unique person(s)")
    return person_counter


if __name__ == "__main__":
    # Example usage
    input_dir = "./my_photos"
    output_dir = "./segregated_faces"
    
    print("Face Segregation Example")
    print("=" * 50)
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print()
    
    # Run segregation
    # num_persons = segregate_faces_example(input_dir, output_dir)
    
    print("To use this example:")
    print("1. Install face_recognition: pip install face_recognition")
    print("2. Create a folder with your images")
    print("3. Run this script or use the CLI tool:")
    print("   face_segregate ./my_photos ./segregated_faces")
