import os

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if not os.path.exists(os.path.join(dirpath, filename)):
                continue
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def get_largest_folders(root_directory, num_folders):
    folder_sizes = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        
        total_size = get_directory_size(dirpath)
        folder_sizes.append((dirpath, total_size))

    largest_folders = sorted(folder_sizes, key=lambda x: x[1], reverse=True)[:num_folders]
    return largest_folders

if __name__ == "__main__":
    folder_path = '/Library'
    # entries = os.listdir(folder_path)

    # for entry in entries:
    #     entry_path = os.path.join(folder_path, entry)
    #     print(entry_path)

    
    # # root_directory = input("Enter the root directory path: ")
    num_folders = 20

    largest_folders = get_largest_folders(folder_path, num_folders)

    print(f"\nTop {num_folders} largest folders:")
    for folder_path, folder_size in largest_folders:
        print(f"{folder_path}: {folder_size} bytes")
