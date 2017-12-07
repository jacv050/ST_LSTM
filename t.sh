a=`docker ps -f name=quyou-st-lstm --quiet`
b="H"$a"L"
if [ -n "$a" ]; then
  echo "Not NULL"
fi
echo $a
echo $b
