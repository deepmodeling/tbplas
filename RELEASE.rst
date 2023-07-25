Procedure of preparing release
==============================

1. Commit all changes and push to remote repository.
2. Update version number in setup.py and doc/source/conf.py using
   scripts/set_ver.py.
3. Prepare doc/source/release.rst.
4. Commit changes and push to remote repository with the message
   "preparing for release X.X.X".
5. Run upload.sh to push the source code and documentation to www.tbplas.net.
6. Update commit id and md5sum in version.
