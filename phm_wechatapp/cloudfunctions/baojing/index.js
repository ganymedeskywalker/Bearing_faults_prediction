//baojing
// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init()

const db = cloud.database();
const _ = db.command;

// 云函数入口函数
exports.main = async (event, context) => {
  return db.collection('machineData').where({
   
    label: event.label
  })
    .get() //get方法查询符合条件的记录，并通过return返回给调用此云函数的代码。
}