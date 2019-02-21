# -*- coding: utf-8 -*-
# @Time    : 2019/2/21 5:35 PM
# @Author  : scl
# @Email   : 1163820757@qq.com
# @File    : testmysql.py
# @Software: PyCharm

'''
 测试连接本地数据库
 参考资料:https://blog.csdn.net/kuangdacaikuang/article/details/76515985
 https://pypi.org/project/PyMySQL/

 终端输入 python3 -m pip install PyMySQL 安装 pymysql模块
'''
import pymysql

# 1 连接本地数据库

connect = pymysql.connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    password = 'long1234',
    database = 'pysqltest',
    charset='utf8'
)

# 2 获取一个光标
cursor = connect.cursor()

'''
插入多条数据
'''
# 3 执行sql 增加一条数据
moresql = 'insert into userinfo(username,age) VALUES(%s,%s);'

data = [
    ('july','1'),
    ('hh','12'),
    ('cheng','23')
]

# 执行sql 语句
cursor.executemany(moresql,data)

# 提交操作
connect.commit()

# 关闭连接
# cursor.close()
# connect.close()



'''
插入单条数据
'''
singlesql = 'insert INTO userinfo(username,age) VALUES (%s,%s);'

cursor.execute(singlesql,['long',99])

connect.commit()

# 关闭连接
# cursor.close()
# connect.close()


'''
获取最新插入的一条数据的id
'''
last_id = cursor.lastrowid
print('最后一条数据的id是:',last_id)
# # 关闭连接
# cursor.close()
# connect.close()


'''
 删除操作
'''
deletesql = 'delete from userinfo WHERE userid=%s;'


userid = 26
cursor.execute(deletesql,[userid])
connect.commit()

#  关闭连接
# cursor.close()
# connect.close()


'''
修改数据 将age替换掉
'''
updatesql = 'update userinfo SET age=%s WHERE username = %s;'
cursor.execute(updatesql,[100,'july'])

connect.commit()
# 关闭连接
# cursor.close()
# connect.close()


'''
 查询数据
'''
cursor = connect.cursor(cursor=pymysql.cursors.DictCursor) # 返回字典数据类型
fetchsql = 'select userid,username,age FROM userinfo'

cursor.execute(fetchsql)

print('所有数据{}'.format(cursor.fetchall()))
# absolute 光标的绝对位置移动多少位
# cursor.scroll(0, mode="relative")# 光标按照相对位置(当前位置)移动1
print('第一条数据{}'.format(cursor.fetchone()))

# 关闭连接
# cursor.close()
# connect.close()


'''
数据回滚操作
'''
cursor = connect.cursor()

sql1 = 'insert INTO userinfo(username,age) VALUES (%s,%s);'
sql2 = 'insert INTO hobby(id,hobby) VALUES (%s,%s);'

try:
    cursor.execute(sql1,['zhangmeng',109])
    cursor.execute(sql2,['错误的数据id','傻子'])
    connect.commit()
except Exception as  e:
    print(str(e))
    # 有异常回滚数据
    connect.rollback()

# 关闭连接
cursor.close()
connect.close()







