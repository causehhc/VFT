<template>
  <div class="Part1___Wrapper">
    <el-col :span="6" class="logoWrapper">
        <el-link href="/" :underline="false">
          <img src="@/assets/logo.png" class="image_logo">
          Beta
        </el-link>
    </el-col>

    <el-col :span="6">
<!--      <el-link href="/dynamic" :underline="false">动态</el-link>-->
      <router-link to="/dynamic">动态</router-link>
    </el-col>
    <el-col :span="6">
<!--      <el-link href="/export" :underline="false">发现</el-link>-->
      <router-link to="/export">发现</router-link>
    </el-col>
    <el-col :span="6">
<!--      <el-link href="/anls" :underline="false">分析</el-link>-->
      <router-link to="/anls">分析</router-link>
    </el-col>
  </div>
</template>

<script>

export default {
  name: 'Part1',
}
</script>

<style scoped>
.Part1___Wrapper{
  margin-top: 10px;
  display:flex;
  text-align: center;
  justify-content:center;/*主轴上居中*/
  align-items:center;/*侧轴上居中*/
}
.logoWrapper{
  display:flex;
  text-align: center;
  justify-content:center;/*主轴上居中*/
  align-items:center;/*侧轴上居中*/
}
.image_logo{
  height: 30px;
  width: 30px;
  position: absolute;
  right: 30px;
  top: -40%;
}
</style>
