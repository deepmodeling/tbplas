#! /bin/bash

# Upload documentation and source code to the official site

function msg ()
{
    echo -e "\033[31;1m$1\033[0m"
}

top_dir=$(pwd)
nginx_dir=/usr/share/nginx

# Generate doc
msg "Generating documentation"
cd $top_dir/doc
make clean
make html
./patch_html.py
msg "Documentation ready"

# Upload doc
msg "Uploading documentation"
cd $top_dir/doc/build
tar -cjf html.tar.bz2 html
scp html.tar.bz2 aliyun:$nginx_dir
ssh aliyun "cd $nginx_dir; rm -rf html; tar -xf html.tar.bz2; ./install_attach.sh; rm -f html.tar.bz2"
rm html.tar.bz2
msg "Done uploading"

# Update src
msg "Uploading source code"
cd $top_dir
./scripts/pack.sh
scp tbplas.tar.* aliyun:$nginx_dir/attachments
rm tbplas.tar.*
msg "Done uploading"
