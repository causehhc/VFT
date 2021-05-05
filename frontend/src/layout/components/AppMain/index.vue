<template>
  <div class="AppMain_Wrapper">
    <el-col class="router" :span="19">
      <router-view :key="key" />
    </el-col>
    <el-col class="sidebar" :span="5">
      <UserSidebar/>
    </el-col>
  </div>
</template>

<script>
import UserSidebar from "./UserSidebar";

export default {
  name: "index",
  // eslint-disable-next-line vue/no-unused-components
  components: {UserSidebar},
  computed: {
    key() {
      return this.$route.path
    }
  }
}
</script>

<style scoped>
.AppMain_Wrapper{
  /*width: 865px;*/
  margin:0 auto;
}
</style>
