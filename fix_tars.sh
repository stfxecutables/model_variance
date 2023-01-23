mkdir -p fixed
# for x in $(fd .tar.gz);
FIXED=$(readlink -f fixed)
cd "$FIXED" || exit 1
for x in $(cat ../tars.txt); do
    parent=$(dirname "$x");
    tarfile=$(basename "$x");
    tarstem="${tarfile//.tar.gz/}";
    newdir=$(basename "$parent")
    mkdir -p "$newdir/$tarstem";
    tar -xzf "$x" --exclude 'dl_logs' -C "$newdir/$tarstem";
    tar -czf "$tarfile" "$newdir/$tarstem";
    rm -rf "$newdir";
done;
