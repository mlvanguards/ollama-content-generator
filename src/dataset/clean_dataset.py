import json
import os
import re


def convert_unicode_escapes(input_file, output_file):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Processing file: {input_file}")

    # Read the JSON file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            json_str = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # First, try to fix any invalid control characters in the JSON
    # Replace common problematic control characters
    cleaned_str = re.sub(r"[\x00-\x09\x0B\x0C\x0E-\x1F]", "", json_str)

    # Try different approaches to parse the JSON
    json_data = None

    # Approach 1: Direct parsing
    try:
        json_data = json.loads(cleaned_str)
        print("JSON parsed successfully without decoding")
    except json.JSONDecodeError as e:
        print(f"Direct parsing failed: {e}")

        # Approach 2: Try to extract the JSON structure manually
        try:
            # Look for the main JSON structure pattern
            json_match = re.search(r"(\{[\s\S]*\})", cleaned_str)
            if json_match:
                extracted_json = json_match.group(1)
                json_data = json.loads(extracted_json)
                print("JSON extracted and parsed successfully")
            else:
                print("Could not extract JSON structure")
        except Exception as e:
            print(f"JSON extraction failed: {e}")

            # Approach 3: Try to fix common JSON syntax issues
            try:
                # Replace problematic Unicode escapes
                fixed_str = re.sub(
                    r"\\u([dD][89abAB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})",
                    lambda m: chr(int(f"0x{m.group(1)}{m.group(2)}", 16))
                    if 0x10000 <= int(f"0x{m.group(1)}{m.group(2)}", 16) <= 0x10FFFF
                    else f"\\u{m.group(1)}\\u{m.group(2)}",
                    cleaned_str,
                )

                # Fix unescaped quotes and other common issues
                fixed_str = fixed_str.replace('\\"', '"').replace("\\\\", "\\")

                json_data = json.loads(fixed_str)
                print("JSON parsed after fixing syntax issues")
            except Exception as e:
                print(f"All JSON parsing attempts failed: {e}")

                # Last resort: Save the cleaned string to a text file for manual inspection
                with open(
                    f"{output_file}.txt", "w", encoding="utf-8", errors="replace"
                ) as f:
                    f.write(cleaned_str)
                print(f"Saved cleaned text to {output_file}.txt for manual inspection")
                return

    if json_data:
        # Process the data to ensure all Unicode is properly decoded
        try:
            # Write the processed JSON with proper indentation
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"Successfully processed and saved to {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")

            # Try writing with ASCII escapes as fallback
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=True, indent=4)
                print(f"Saved with ASCII escapes to {output_file}")
            except Exception as e:
                print(f"All writing attempts failed: {e}")


if __name__ == "__main__":
    input_file = "data/posts_test_data.json"
    output_file = "data/posts_test_data_decoded.json"

    convert_unicode_escapes(input_file, output_file)
