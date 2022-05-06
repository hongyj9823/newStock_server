import re

string = '곽경택 감독 첫 오디오무비 \'극동\', <b>네이버</b> 바이브 공개[공식]'

output = re.sub(" <b>.*?</b>", "", string)

print(output)