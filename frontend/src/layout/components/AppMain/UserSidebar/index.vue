<template>
  <div class="UserSidebar_Wrapper">
    <el-row>
      <Part1/>
    </el-row>
    <el-row>
      <Part2/>
    </el-row>
    <el-row>
      <Part3/>
    </el-row>
  </div>
</template>

<script>
import Part1 from "./Part1";
import Part2 from "./Part2";
import Part3 from "./Part3";

export default {
  name: "index",
  components: {Part1, Part2, Part3}
}
</script>

<style scoped>
.UserSidebar_Wrapper{
  /*text-align: center;*/
  /*margin-top: 14px;*/
  /*background: #bff2f2;*/
}
</style>
