<template>
  <div class="Part2___Wrapper">
    <el-input v-model="input" placeholder="搜索">
      <i slot="prefix" class="el-input__icon el-icon-search"></i>
    </el-input>
  </div>
</template>

<script>
export default {
  name: "Part2",
  data() {
    return {
      input: ''
    }
  }
}
</script>

<style scoped>
.Part2___Wrapper{
  margin-top: 10px;
  /*height: 10px;*/
}
</style>
