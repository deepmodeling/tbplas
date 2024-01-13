#! /bin/bash
# Pack source code for distribution

proj_dir=$HOME/proj/tbplas
items=$(ls $proj_dir)  # It must be placed here!
top_dir=$(pwd)

# Copy items to destination
cd $top_dir
test -d tmp/tbplas && rm -rf tmp/tbplas
mkdir -p tmp/tbplas
for item in $items; do
    cp -r $proj_dir/$item tmp/tbplas
done

# Remove unnecessary files
cd $top_dir/tmp/tbplas
for kind in pyc so; do
    find . -name "*.$kind" | xargs rm -rf
done
for kind in __pycache__ sim_data build; do
    find . -name "$kind" | xargs rm -rf
done

# Compress
cd $top_dir/tmp
tar -cjf tbplas.tar.bz2 tbplas
mv tbplas.tar.bz2 ..

# Clean
cd $top_dir
rm -rf tmp
