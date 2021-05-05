<template>
  <div id="app" class="force" ref="element">
<!--    <button @click="tempClick"/>-->
    <network
        :nodeList="nodes"
        :linkList="links"
        :svgTheme="svgTheme"
        :svgSize="svgSize"
    />
  </div>
</template>

<script>
import Network from "vue-network-d3";
import {getGraph} from "@/api/anls";

export default {
  name: "app",
  components: {
    Network
  },
  data() {
    return {
      nodes: [],
      links: [],
      svgSize: {
        width: 1100,
        height: 525,
        // width: this.$refs['element'].clientWidth,
        // height: this.$refs['element'].clientHeight
      },
      svgTheme: "light",
    };
  },
  created() {
    this.fetchData()
  },
  methods:{
    tempClick(){
      console.log(this.$refs['element'].clientWidth)
      console.log(this.$refs['element'].clientHeight)
      console.log(this.svgSize)
    },
    fetchData(){
      setTimeout(() => {
        getGraph({
          token: this.$store.getters.token
        }).then(response => {
          this.nodes = response.data.nodes;
          this.links = response.data.links;
        });
      }, 10)
    }
  }
};
</script>

<style>
body {
  margin: 0;
}
</style>
