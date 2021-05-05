<template>
  <div class="NavBar_Wrapper">
    <el-col :span="12">
      <Part1/>
    </el-col>
    <el-col :span="8">
      <Part2/>
    </el-col>
    <el-col :span="4">
      <Part3/>
    </el-col>
  </div>
</template>

<script>
import Part1 from "./Part1";
import Part2 from "./Part2";
import Part3 from "./Part3";

export default {
  name: "index",
  components: {Part1, Part2, Part3},
}
</script>

<style scoped>
.NavBar_Wrapper{
  display:flex;
  text-align: center;
  /*justify-content:center;!*主轴上居中*!*/
  align-items:center;/*侧轴上居中*/
}
</style>
