# How to load big file

git init
git add .
git commit -m "initial upload"

To publish this branch directly in github just go to left tab click the 3 github sign then there is a publish/commit button then the repo will be published.

then add the big file to the git folder then type the following command

git lfs install
git lfs tack "bigfile.h5"
git add .gitattributes
git add bigfile.h5
git commit -m "commit message"
git push origin master