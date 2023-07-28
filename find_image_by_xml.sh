#!bin/bash
# 查找并解压 zip 文件

# 检查参数个数
if [ "$#" -ne 1 ]; then
  echo "错误：需要传入两个参数。"
  echo "使用方法：$0 \$1 \$2"
  echo "\$1: 指定目录，用于查找 zip 文件和 xml 文件。"
  exit 1
fi

# 检查目录是否存在
if [ ! -d "$1" ]; then
  echo "错误：目录 $1 不存在。"
  exit 1
fi

# xml压缩文件的目录
xml_zip_dir=$1
# 临时解压到指定目录下
out_xml_dir="/opt/gongdi/biaozhu/image/auto_biaozhu/out_xml/"
# 搜索到的图片存放指定目录内
out_image_path="/opt/gongdi/biaozhu/image/auto_biaozhu/out_image/"

mkdir -p $out_xml_dir
mkdir -p $out_image_path

rm -rf $out_xml_dir/*

find "$xml_zip_dir" -type f -name "*.zip" -exec unzip {} -d "$out_xml_dir" \;


rm -rf $out_image_path/*.*

# 从/opt/gongdi/biaozhu/1080P/目录下查找所有.xml文件，截取文件名，不带目录和后缀
for filepath in $out_xml_dir*.xml; do
    filename=$(basename "$filepath" .xml)
    echo "File name is: $filename"
    
    # 在/opt/gongdi/copy0308/output_1080/目录下查找对应的.jpg文件
    result=$(find /opt/gongdi/copy0308/output_1080/ -name "$filename.jpg")
    
    # 打印查找结果
    if [ -z "$result" ]; then
        echo "No matching .jpg file found for $filename"
    else
        echo "Matching .jpg file(s) found for $filename:"
        echo "$result"
        
        # 将所有找到的.jpg文件拷贝到$$out_image_path目录下，并打印拷贝结果
        cp $result $out_image_path
        echo "Matching .jpg file(s) coped to " $out_image_path
    fi
    
done
