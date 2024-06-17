from modal import Image

image = (
    Image.debian_slim(python_version='3.11')
    .apt_install('git')
    .run_commands("git clone ")
)