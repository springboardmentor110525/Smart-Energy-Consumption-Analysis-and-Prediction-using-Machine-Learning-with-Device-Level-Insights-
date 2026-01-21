@echo off
cd /d "d:\Aditya Anand\Aditya_Anand"
git config user.email "bot@antigravity.com"
git config user.name "Antigravity Bot"
git checkout -b Aditya_Anand
git add .
git commit -m "Add project files assignment"
git push -u origin Aditya_Anand
git status > status_log.txt
echo DONE
