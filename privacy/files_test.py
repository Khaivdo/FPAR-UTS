import os

test_dir = './test_dir'
output_dir = './output_dir'

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# os.path.join(video_dir, filename)

if __name__ == '__main__':
    create_dir_if_not_exists(output_dir)
    for dirpath, dirnames, filenames in os.walk(test_dir):
        new_dirpath = dirpath.replace(test_dir, output_dir)
        print(dirpath, filenames)
        print(new_dirpath)
        create_dir_if_not_exists(new_dirpath)
