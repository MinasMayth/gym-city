while read -r line; do
    eval "$line"
done < commands.txt
