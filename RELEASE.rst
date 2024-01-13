Procedure of preparing release
==============================

1. Commit all changes and push to remote repository.
2. Update APIs in doc/source/api.rst using scripts/get_api.py.
3. Update version number and date using scripts/update_doc.py.
4. Prepare doc/source/release.rst.
5. Commit changes and push to remote repository with the message
   "prepare for release X.X.X".
6. Run upload.sh to push the source code and documentation to www.tbplas.net.
7. Update commit id and md5sum in version.
