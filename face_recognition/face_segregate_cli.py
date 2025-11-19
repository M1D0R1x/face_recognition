# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import numpy as np


def image_files_in_folder(folder):
    """Get all image files in a folder"""
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def extract_and_save_face(image_path, face_location, output_path):
    """Extract a face from an image and save it to a file"""
    image = face_recognition.load_image_file(image_path)
    top, right, bottom, left = face_location
    
    # Extract the face
    face_image = image[top:bottom, left:right]
    
    # Save the face image
    from PIL import Image
    pil_image = Image.fromarray(face_image)
    pil_image.save(output_path)


def segregate_faces(input_folder, output_folder, tolerance=0.6, model="hog"):
    """
    Segregate faces from images in input_folder into separate folders in output_folder.
    
    :param input_folder: Folder containing images to process
    :param output_folder: Folder where segregated faces will be saved
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict.
    :param model: Which face detection model to use. "hog" or "cnn"
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Dictionary to store known face encodings and their person IDs
    known_face_encodings = []
    person_ids = []
    person_counter = 0
    
    # Dictionary to track which person folder to save to
    person_folders = {}
    
    # Get all image files
    image_files = image_files_in_folder(input_folder)
    
    if not image_files:
        click.echo(f"No image files found in {input_folder}")
        return
    
    click.echo(f"Processing {len(image_files)} images...")
    
    # Process each image
    for image_path in image_files:
        click.echo(f"Processing {os.path.basename(image_path)}...")
        
        # Load image and find faces
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            click.echo(f"  No faces found in {os.path.basename(image_path)}")
            continue
        
        click.echo(f"  Found {len(face_encodings)} face(s)")
        
        # Process each face in the image
        for face_idx, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            # Compare with known faces
            if len(known_face_encodings) == 0:
                # First face found
                person_id = person_counter
                person_counter += 1
                known_face_encodings.append(face_encoding)
                person_ids.append(person_id)
            else:
                # Compare with existing faces
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                
                if distances[best_match_index] <= tolerance:
                    # Match found
                    person_id = person_ids[best_match_index]
                else:
                    # New person
                    person_id = person_counter
                    person_counter += 1
                    known_face_encodings.append(face_encoding)
                    person_ids.append(person_id)
            
            # Create person folder if it doesn't exist
            person_folder_name = f"person_{person_id}"
            person_folder_path = os.path.join(output_folder, person_folder_name)
            
            if not os.path.exists(person_folder_path):
                os.makedirs(person_folder_path)
                person_folders[person_id] = person_folder_path
            
            # Save the face image
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            face_filename = f"{image_basename}_face_{face_idx}.jpg"
            face_output_path = os.path.join(person_folder_path, face_filename)
            
            extract_and_save_face(image_path, face_location, face_output_path)
            click.echo(f"  Saved face to {person_folder_name}/{face_filename}")
    
    click.echo(f"\nSegregation complete! Found {person_counter} unique person(s).")
    click.echo(f"Results saved to: {output_folder}")
    
    # Print summary
    for person_id in range(person_counter):
        person_folder = os.path.join(output_folder, f"person_{person_id}")
        num_faces = len(os.listdir(person_folder))
        click.echo(f"  person_{person_id}: {num_faces} face image(s)")


@click.command()
@click.argument('input_folder')
@click.argument('output_folder')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple persons for the same face.')
@click.option('--model', default="hog", help='Which face detection model to use. Options are "hog" or "cnn".')
def main(input_folder, output_folder, tolerance, model):
    """
    Segregate faces from images in INPUT_FOLDER into separate person folders in OUTPUT_FOLDER.
    
    This tool will:
    1. Detect all faces in the input images
    2. Group similar faces together using face recognition
    3. Save each person's face images into separate folders (person_0, person_1, etc.)
    
    Example usage:
    
        face_segregate ./my_images ./segregated_faces
    
    You can adjust the tolerance to make face matching more or less strict:
    
        face_segregate ./my_images ./segregated_faces --tolerance 0.5
    """
    if not os.path.isdir(input_folder):
        click.echo(f"Error: {input_folder} is not a valid directory")
        return
    
    segregate_faces(input_folder, output_folder, tolerance, model)


if __name__ == "__main__":
    main()
