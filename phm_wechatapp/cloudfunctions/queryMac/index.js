//查询机组功能
// 云函数入口文件
const cloud = require('wx-server-sdk')
cloud.init()
const db = cloud.database();

// 云函数入口函数
exports.main = async (event, context) => {
  return db.collection(event.set).where({
    date:event.date,
    predict:event.predict,
    principal:event.principal,
  })
    .get() //get方法查询符合条件的记录，并通过return返回给调用此云函数的代码。
}